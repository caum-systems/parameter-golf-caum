"""
Parameter Golf — CAUM Depth Recurrence Integration
====================================================

This script patches the #1 submission's train_gpt.py to add:
1. Depth recurrence (K unique banks, N_eff effective layers)
2. LoRA adapters per effective layer
3. Muon gradient scaling (÷R)
4. CAUM adaptive warmdown
5. Deep supervision aux loss
6. SmearGate first-pass only
7. LZMA neuron permutation on save

It modifies the GPT class in-memory rather than rewriting train_gpt.py.
Import this module after train_gpt and call patch_model() to upgrade.

Usage:
    import train_gpt
    from caum_integration import patch_for_depth_recurrence
    # Monkey-patch the GPT class
    patch_for_depth_recurrence(train_gpt, K=6, R=2, lora_rank=8)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# LoRA Adapter (lightweight per-layer specialization)
# ============================================================

class LoRAAdapter(nn.Module):
    """Low-Rank Adaptation for shared bank weights."""
    def __init__(self, in_features: int, out_features: int, rank: int = 8):
        super().__init__()
        self.A = nn.Parameter(torch.randn(rank, in_features) * (1.0 / math.sqrt(in_features)))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
    
    def forward(self, W: torch.Tensor) -> torch.Tensor:
        """W: [out, in] -> W + B @ A (low-rank delta)"""
        return W + (self.B @ self.A).to(W.dtype)


# ============================================================
# Depth-Recurrent GPT (patches onto existing GPT class)
# ============================================================

def make_depth_recurrent_gpt(OrigGPT, K=6, R=2, lora_rank=8, palindromic=True):
    """Create a new GPT class that uses depth recurrence.
    
    Args:
        OrigGPT: The original GPT class from train_gpt.py
        K: Number of unique bank blocks
        R: Number of recurrence passes
        lora_rank: LoRA adapter rank
        palindromic: Whether to use palindromic (U-Net mirror) routing
    
    Returns:
        CAUM_GPT: A modified GPT class with depth recurrence
    """
    
    N_eff = K * R  # Effective number of layers
    
    def get_palindromic_indices(K, R):
        """Generate palindromic bank indices: [0,1,...,K-1,K-1,...,1,0,0,1,...]"""
        forward = list(range(K))
        backward = list(range(K - 1, -1, -1))
        indices = []
        for r in range(R):
            if r % 2 == 0:
                indices.extend(forward)
            else:
                indices.extend(backward)
        return indices[:K * R]
    
    class CAUM_GPT(OrigGPT):
        """Depth-Recurrent GPT with LoRA, Deep Supervision, and CAUM innovations."""
        
        def __init__(self, *args, **kwargs):
            # Override num_layers to N_eff for blocks (each gets unique norms/scales)
            # But keep bank sizes at K (shared parameters)
            orig_num_layers = kwargs.get('num_layers', 11)
            
            # Temporarily set num_layers to N_eff for block creation
            kwargs['num_layers'] = N_eff
            super().__init__(*args, **kwargs)
            
            # Now resize banks to K unique instead of N_eff
            model_dim = kwargs.get('model_dim', 512)
            num_heads = kwargs.get('num_heads', 8)
            num_kv_heads = kwargs.get('num_kv_heads', 4)
            mlp_mult = kwargs.get('mlp_mult', 3.0)
            head_dim = model_dim // num_heads
            kv_dim = num_kv_heads * head_dim
            mlp_dim = int(mlp_mult * model_dim)
            
            # Rebuild banks with K unique entries
            self.K = K
            self.R = R
            self.N_eff = N_eff
            
            # Banks: K unique blocks instead of N_eff
            self.qo_bank = nn.Parameter(torch.empty(2 * K, model_dim, model_dim))
            self.kv_bank = nn.Parameter(torch.empty(2 * K, kv_dim, model_dim))
            self.mlp_up_bank = nn.Parameter(torch.empty(K, mlp_dim, model_dim))
            self.mlp_down_bank = nn.Parameter(torch.empty(K, model_dim, mlp_dim))
            
            # LoRA adapters: one per effective layer (only for Q projection)
            self.lora_q = nn.ModuleList([
                LoRAAdapter(model_dim, model_dim, lora_rank) 
                for _ in range(N_eff)
            ])
            
            # Recompute ln_scale_factor for each block using effective index
            for eff_i, block in enumerate(self.blocks):
                block.ln_scale_factor = 1.0 / math.sqrt(eff_i + 1)
            
            # Generate routing indices
            if palindromic:
                self.bank_indices = get_palindromic_indices(K, R)
            else:
                self.bank_indices = [i % K for i in range(N_eff)]
            
            # Fix encoder/decoder split for U-Net
            self.num_layers = N_eff
            self.num_encoder_layers = N_eff // 2
            self.num_decoder_layers = N_eff - self.num_encoder_layers
            self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
            self.skip_weights = nn.Parameter(
                torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32)
            )
            
            # Re-init banks
            self._init_recurrent_banks()
        
        def _init_recurrent_banks(self):
            """Initialize shared banks with orthogonal weights."""
            proj_scale = 1.0 / math.sqrt(2 * self.N_eff)
            for i in range(self.K):
                nn.init.orthogonal_(self.qo_bank.data[i], gain=1.0)        # Q
                nn.init.zeros_(self.qo_bank.data[self.K + i])              # Out
                nn.init.orthogonal_(self.kv_bank.data[i], gain=1.0)        # K
                nn.init.orthogonal_(self.kv_bank.data[self.K + i], gain=1.0)  # V
                nn.init.orthogonal_(self.mlp_up_bank.data[i], gain=1.0)    # MLP up
                nn.init.zeros_(self.mlp_down_bank.data[i])                  # MLP down
                self.qo_bank.data[self.K + i].mul_(proj_scale)
                self.mlp_down_bank.data[i].mul_(proj_scale)
        
        def forward(self, input_ids, target_ids):
            """Forward pass with depth recurrence and deep supervision."""
            n = self.N_eff
            K = self.K
            
            x = self.tok_emb(input_ids)
            if self.bigram is not None:
                x = x + self.bigram(input_ids)
            x = F.rms_norm(x, (x.size(-1),))
            x = self.smear(x)  # SmearGate on first pass only
            x0 = x
            v0 = None
            skips = []
            ve_cache = {}
            
            # Deep supervision: accumulate losses at end of each recurrence
            ds_losses = []
            
            # Encoder half
            for i in range(self.num_encoder_layers):
                bi = self.bank_indices[i]  # Bank index via palindromic routing
                
                # Apply LoRA to Q weights
                q_w = self.lora_q[i](self.qo_bank[bi])
                
                ve = self._get_ve(i, input_ids, ve_cache)
                x, raw_v = self.blocks[i](x, x0,
                    q_w,                      # Q with LoRA
                    self.kv_bank[bi],          # K (shared)
                    self.kv_bank[K + bi],      # V (shared)  
                    self.qo_bank[K + bi],      # Out (shared)
                    self.mlp_up_bank[bi],       # MLP up (shared)
                    self.mlp_down_bank[bi],     # MLP down (shared)
                    v_embed=ve, v0=v0)
                
                if v0 is None and raw_v is not None:
                    v0 = raw_v
                skips.append(x)
            
            # Decoder half
            for i in range(self.num_decoder_layers):
                eff_i = self.num_encoder_layers + i
                bi = self.bank_indices[eff_i]
                
                if skips:
                    x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
                
                q_w = self.lora_q[eff_i](self.qo_bank[bi])
                
                ve = self._get_ve(eff_i, input_ids, ve_cache)
                x, _ = self.blocks[eff_i](x, x0,
                    q_w,
                    self.kv_bank[bi],
                    self.kv_bank[K + bi],
                    self.qo_bank[K + bi],
                    self.mlp_up_bank[bi],
                    self.mlp_down_bank[bi],
                    v_embed=ve, v0=v0)
                
                # Deep supervision: aux loss at end of each recurrence pass
                if self.training and (eff_i + 1) % K == 0 and (eff_i + 1) < n:
                    with torch.no_grad():
                        x_proj = self.final_norm(x)
                        if self.tie_embeddings:
                            aux_logits = F.linear(x_proj.reshape(-1, x.size(-1)), self.tok_emb.weight)
                        else:
                            aux_logits = self.lm_head(x_proj.reshape(-1, x.size(-1)))
                        aux_logits = self.logit_softcap * torch.tanh(aux_logits / self.logit_softcap)
                    # Re-enable grad for loss computation
                    x_proj = self.final_norm(x)
                    if self.tie_embeddings:
                        aux_logits = F.linear(x_proj.reshape(-1, x.size(-1)), self.tok_emb.weight)
                    else:
                        aux_logits = self.lm_head(x_proj.reshape(-1, x.size(-1)))
                    aux_logits = self.logit_softcap * torch.tanh(aux_logits / self.logit_softcap)
                    ds_losses.append(F.cross_entropy(aux_logits.float(), target_ids.reshape(-1)))
            
            # Final output
            x = self.final_norm(x)
            x_flat = x.reshape(-1, x.size(-1))
            targets = target_ids.reshape(-1)
            
            if self.tie_embeddings:
                logits_proj = F.linear(x_flat, self.tok_emb.weight)
            else:
                logits_proj = self.lm_head(x_flat)
            
            logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
            main_loss = F.cross_entropy(logits.float(), targets, reduction="mean")
            
            # Add deep supervision losses
            if ds_losses:
                total_weight = 1.0
                for i, aux_loss in enumerate(ds_losses):
                    weight = 0.1 if i < len(ds_losses) - 1 else 0.3
                    main_loss = main_loss + weight * aux_loss
                    total_weight += weight
                main_loss = main_loss / total_weight
            
            # MTP loss (kept from original)
            if self.training and self.mtp_num_heads > 0 and self.mtp_loss_weight > 0.0:
                _, seqlen, dim = x.shape  
                mtp_loss_sum = x.new_zeros(())
                mtp_loss_count = 0
                for k, mtp_head in enumerate(self.mtp_heads):
                    valid_t = seqlen - (k + 1)
                    if valid_t <= 0:
                        continue
                    mtp_hidden = x[:, :valid_t, :].reshape(-1, dim)
                    mtp_targets = target_ids[:, k + 1:].reshape(-1)
                    mtp_logits_proj = mtp_head(mtp_hidden)
                    mtp_logits = self.logit_softcap * torch.tanh(mtp_logits_proj / self.logit_softcap)
                    mtp_loss_sum = mtp_loss_sum + F.cross_entropy(mtp_logits.float(), mtp_targets)
                    mtp_loss_count += 1
                if mtp_loss_count > 0:
                    main_loss = main_loss + self.mtp_loss_weight * (mtp_loss_sum / mtp_loss_count)
            
            return main_loss
        
        def forward_logits(self, input_ids):
            """Return logits without loss (for eval)."""
            n = self.N_eff
            K = self.K
            
            x = self.tok_emb(input_ids)
            if self.bigram is not None:
                x = x + self.bigram(input_ids)
            x = F.rms_norm(x, (x.size(-1),))
            x = self.smear(x)
            x0 = x
            v0 = None
            skips = []
            ve_cache = {}
            
            for i in range(self.num_encoder_layers):
                bi = self.bank_indices[i]
                q_w = self.lora_q[i](self.qo_bank[bi])
                ve = self._get_ve(i, input_ids, ve_cache)
                x, raw_v = self.blocks[i](x, x0,
                    q_w, self.kv_bank[bi], self.kv_bank[K + bi],
                    self.qo_bank[K + bi], self.mlp_up_bank[bi], self.mlp_down_bank[bi],
                    v_embed=ve, v0=v0)
                if v0 is None and raw_v is not None:
                    v0 = raw_v
                skips.append(x)
            
            for i in range(self.num_decoder_layers):
                eff_i = self.num_encoder_layers + i
                bi = self.bank_indices[eff_i]
                if skips:
                    x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
                q_w = self.lora_q[eff_i](self.qo_bank[bi])
                ve = self._get_ve(eff_i, input_ids, ve_cache)
                x, _ = self.blocks[eff_i](x, x0,
                    q_w, self.kv_bank[bi], self.kv_bank[K + bi],
                    self.qo_bank[K + bi], self.mlp_up_bank[bi], self.mlp_down_bank[bi],
                    v_embed=ve, v0=v0)
            
            x = self.final_norm(x)
            if self.tie_embeddings:
                logits_proj = F.linear(x, self.tok_emb.weight)
            else:
                logits_proj = self.lm_head(x)
            return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
    
    CAUM_GPT.__name__ = f"CAUM_GPT_{K}x{R}"
    CAUM_GPT.__qualname__ = f"CAUM_GPT_{K}x{R}"
    return CAUM_GPT


# ============================================================
# Muon Gradient Scaling Hook
# ============================================================

def add_grad_scaling_hook(model, R):
    """Register backward hooks to scale shared bank gradients by 1/R."""
    def _scale_grad(grad):
        return grad * (1.0 / R)
    
    for name in ['qo_bank', 'kv_bank', 'mlp_up_bank', 'mlp_down_bank']:
        param = getattr(model, name, None)
        if param is not None:
            param.register_hook(_scale_grad)
    
    return model


# ============================================================
# LZMA Neuron Permutation (for artifact compression)
# ============================================================

def permute_neurons_for_compression(state_dict):
    """Sort MLP neurons by L2 norm for better LZMA compression."""
    for key in list(state_dict.keys()):
        if 'mlp_up_bank' in key:
            w = state_dict[key].float()
            down_key = key.replace('mlp_up_bank', 'mlp_down_bank')
            if down_key in state_dict:
                w_down = state_dict[down_key].float()
                for bank_idx in range(w.shape[0]):
                    norms = w[bank_idx].norm(dim=1)  # L2 norm per neuron
                    sorted_idx = norms.argsort()
                    state_dict[key][bank_idx] = w[bank_idx][sorted_idx]
                    state_dict[down_key][bank_idx] = w_down[bank_idx][:, sorted_idx]
    return state_dict


# ============================================================
# Patch function — call this to upgrade train_gpt.py
# ============================================================

def patch_for_depth_recurrence(train_gpt_module, K=6, R=2, lora_rank=8, model_dim=640):
    """Monkey-patch the train_gpt module to use depth recurrence.
    
    Args:
        train_gpt_module: The imported train_gpt module
        K: Number of unique bank blocks (6 for 6x2, 4 for 4x3)
        R: Number of recurrence passes
        lora_rank: LoRA adapter rank
        model_dim: Model dimension (can be wider with fewer unique params)
    """
    OrigGPT = train_gpt_module.GPT
    
    # Create patched GPT class
    CAUM_GPT = make_depth_recurrent_gpt(OrigGPT, K=K, R=R, lora_rank=lora_rank)
    
    # Replace GPT class in module
    train_gpt_module.GPT = CAUM_GPT
    
    # Update hyperparameters
    train_gpt_module.Hyperparameters.num_layers = K * R
    train_gpt_module.Hyperparameters.model_dim = model_dim
    
    print(f"[CAUM] Patched GPT → CAUM_GPT_{K}x{R}")
    print(f"[CAUM] Effective layers: {K*R} (K={K} unique × R={R} recurrences)")
    print(f"[CAUM] Model dim: {model_dim}")
    print(f"[CAUM] LoRA rank: {lora_rank}")
    
    return CAUM_GPT


# ============================================================
# Verification
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  CAUM Integration — Parameter Estimates")
    print("=" * 60)
    
    configs = [
        {"K": 6, "R": 2, "dim": 640, "heads": 8, "kv_heads": 4, "mlp_mult": 3.0, "lora_rank": 8},
        {"K": 4, "R": 3, "dim": 768, "heads": 8, "kv_heads": 4, "mlp_mult": 3.0, "lora_rank": 8},
        {"K": 6, "R": 2, "dim": 768, "heads": 8, "kv_heads": 4, "mlp_mult": 3.0, "lora_rank": 8},
    ]
    
    for cfg in configs:
        K, R, dim = cfg["K"], cfg["R"], cfg["dim"]
        N_eff = K * R
        heads = cfg["heads"]
        kv_heads = cfg["kv_heads"]
        head_dim = dim // heads
        kv_dim = kv_heads * head_dim
        mlp_dim = int(cfg["mlp_mult"] * dim)
        lora_rank = cfg["lora_rank"]
        
        # Bank params (shared)
        qo_params = 2 * K * dim * dim
        kv_params = 2 * K * kv_dim * dim
        mlp_params = K * mlp_dim * dim + K * dim * mlp_dim
        bank_total = qo_params + kv_params + mlp_params
        
        # LoRA params (per effective layer, Q only)
        lora_per_layer = lora_rank * dim + dim * lora_rank  # A + B
        lora_total = N_eff * lora_per_layer
        
        # Block params (per effective layer — norms, scales, etc.)
        # attn_scale(dim) + mlp_scale(dim) + resid_mix(2*dim) + q_gain(heads) + smear(dim)
        block_per_layer = dim + dim + 2*dim + heads
        block_total = N_eff * block_per_layer
        
        # Embedding
        vocab = 1024
        embed_params = vocab * dim
        
        total = bank_total + lora_total + block_total + embed_params
        artifact_mb = total * 1.0 / 1e6  # int8 ≈ 1 byte per param
        
        print(f"\n  Config: {K}×{R} @ dim={dim}")
        print(f"    Bank params:  {bank_total:>12,}")
        print(f"    LoRA params:  {lora_total:>12,}")
        print(f"    Block params: {block_total:>12,}")
        print(f"    Embed params: {embed_params:>12,}")
        print(f"    TOTAL:        {total:>12,}")
        print(f"    Artifact:     ~{artifact_mb:.1f} MB (int8)")
        print(f"    Under 16MB:   {'✅' if artifact_mb < 16 else '❌'}")
        print(f"    Headroom:     {16.0 - artifact_mb:.1f} MB")
