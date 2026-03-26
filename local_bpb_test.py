"""
Parameter Golf — Local BPB Estimation
=======================================
Compares baseline (11-layer, 512d) vs our depth-recurrent fork
on real text data to estimate expected BPB improvement.

Uses WikiText-2 as a proxy (similar to FineWeb distribution).
Runs 500 steps on RTX 4070 SUPER for each config.

Expected output: relative BPB delta between baseline and fork.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depth_recurrence_patch import LoRAAdapter
from optimizations_patch import scale_shared_bank_gradients, get_palindromic_bank_indices

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16 if device == 'cuda' else torch.float32


# ============================================================
# Data: Use a simple character-level dataset for fair comparison
# ============================================================

def get_text_data(seq_len=512, num_seqs=2000):
    """Download and tokenize a text dataset for BPB estimation."""
    # Use a built-in approach: generate tokens from torch's randn 
    # OR try to load real text
    
    data_path = os.path.join(os.path.dirname(__file__), "test_data.txt")
    
    if not os.path.exists(data_path):
        # Download a chunk of text for testing
        print("  Downloading test text data...")
        try:
            import urllib.request
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            urllib.request.urlretrieve(url, data_path)
            print("  Downloaded tinyshakespeare (1.1MB)")
        except Exception as e:
            print(f"  Download failed: {e}")
            # Fallback: generate synthetic text-like data
            print("  Using synthetic data instead")
            return None, None
    
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Character-level tokenization (simple, fair comparison)
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    
    # Split into sequences
    n = len(data) - seq_len - 1
    indices = torch.randint(0, n, (num_seqs,))
    
    x = torch.stack([data[i:i+seq_len] for i in indices])
    y = torch.stack([data[i+1:i+seq_len+1] for i in indices])
    
    return x, y, vocab_size


# ============================================================
# Model A: Standard Transformer (like the baseline)
# ============================================================

class StandardTransformer(nn.Module):
    """Standard N-layer transformer (no weight sharing)."""
    
    def __init__(self, vocab_size, dim, n_layers, n_heads=8):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, i) for i in range(n_layers)
        ])
        self.norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # tie embeddings
        self.dim = dim
    
    def forward(self, x_ids, targets):
        x = self.tok_emb(x_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.head(x)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, layer_idx):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.mlp_up = nn.Linear(dim, dim * 3, bias=False)
        self.mlp_down = nn.Linear(dim * 3, dim, bias=False)
        self.n_heads = n_heads
        self.scale = 1.0 / math.sqrt(layer_idx + 1)
        self.head_dim = dim // n_heads
    
    def forward(self, x):
        B, T, C = x.shape
        # Attention
        h = self.norm1(x)
        q = self.q(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.out(attn) * self.scale
        
        # MLP with LeakyReLU²
        h = self.norm2(x)
        up = self.mlp_up(h)
        up = F.leaky_relu(up, 0.5) ** 2
        x = x + self.mlp_down(up) * self.scale
        return x


# ============================================================
# Model B: Depth-Recurrent Transformer (our fork)
# ============================================================

class RecurrentTransformer(nn.Module):
    """Depth-recurrent transformer with LoRA, palindromic routing, deep supervision."""
    
    def __init__(self, vocab_size, dim, K, R, n_heads=8, lora_rank=8):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.dim = dim
        self.K = K
        self.R = R
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        # Shared banks (K unique blocks)
        self.q_bank = nn.ParameterList([nn.Parameter(torch.randn(dim, dim) * 0.02) for _ in range(K)])
        self.k_bank = nn.ParameterList([nn.Parameter(torch.randn(dim, dim) * 0.02) for _ in range(K)])
        self.v_bank = nn.ParameterList([nn.Parameter(torch.randn(dim, dim) * 0.02) for _ in range(K)])
        self.out_bank = nn.ParameterList([nn.Parameter(torch.randn(dim, dim) * 0.02) for _ in range(K)])
        self.mlp_up_bank = nn.ParameterList([nn.Parameter(torch.randn(dim*3, dim) * 0.02) for _ in range(K)])
        self.mlp_down_bank = nn.ParameterList([nn.Parameter(torch.randn(dim, dim*3) * 0.02) for _ in range(K)])
        
        # Per-position LoRA adapters
        n_eff = K * R
        self.lora_q = nn.ModuleList([LoRAAdapter(dim, dim, lora_rank) for _ in range(n_eff)])
        
        # Per-position norms
        self.norms1 = nn.ModuleList([nn.RMSNorm(dim) for _ in range(n_eff)])
        self.norms2 = nn.ModuleList([nn.RMSNorm(dim) for _ in range(n_eff)])
        
        # Palindromic routing indices
        self.bank_indices = get_palindromic_bank_indices(K, R)
    
    def forward(self, x_ids, targets, deep_supervision=True):
        B, T = x_ids.shape
        x = self.tok_emb(x_ids)
        
        # Deep supervision weights
        ds_weights = []
        for r in range(self.R):
            if r == self.R - 1:
                ds_weights.append(1.0)
            elif r == self.R - 2:
                ds_weights.append(0.3)
            else:
                ds_weights.append(0.1)
        
        total_loss = torch.zeros((), device=x.device, dtype=torch.float32)
        flat_targets = targets.view(-1)
        
        for eff_i, bi in enumerate(self.bank_indices):
            scale = 1.0 / math.sqrt(eff_i + 1)
            
            # Attention
            h = self.norms1[eff_i](x)
            q_w = self.lora_q[eff_i](self.q_bank[bi])
            q = F.linear(h, q_w.to(h.dtype)).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = F.linear(h, self.k_bank[bi].to(h.dtype)).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            v = F.linear(h, self.v_bank[bi].to(h.dtype)).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            attn = attn.transpose(1, 2).contiguous().view(B, T, self.dim)
            x = x + F.linear(attn, self.out_bank[bi].to(h.dtype)) * scale
            
            # MLP with LeakyReLU²
            h = self.norms2[eff_i](x)
            up = F.linear(h, self.mlp_up_bank[bi].to(h.dtype))
            up = F.leaky_relu(up, 0.5) ** 2
            x = x + F.linear(up, self.mlp_down_bank[bi].to(h.dtype)) * scale
            
            # Deep supervision: compute loss at end of each recurrence pass
            if deep_supervision and (eff_i + 1) % self.K == 0:
                rep = (eff_i + 1) // self.K - 1
                x_proj = self.norm(x)
                logits = self.head(x_proj).view(-1, self.head.out_features)
                pass_loss = F.cross_entropy(logits.float(), flat_targets)
                total_loss = total_loss + ds_weights[rep] * pass_loss
        
        if not deep_supervision:
            x = self.norm(x)
            logits = self.head(x).view(-1, self.head.out_features)
            return F.cross_entropy(logits.float(), flat_targets)
        
        return total_loss / sum(ds_weights)


# ============================================================
# Training + Evaluation Loop
# ============================================================

def train_and_evaluate(model, train_x, train_y, val_x, val_y, name, steps=500, batch_size=16, lr=1e-3):
    """Train for N steps, return final validation loss and BPB."""
    model = model.to(device).to(dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    
    n_params = sum(p.numel() for p in model.parameters())
    n_unique = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Params: {n_params:,} | Trainable: {n_unique:,}")
    print(f"{'='*60}")
    
    model.train()
    losses = []
    t0 = time.time()
    
    n_train = train_x.shape[0]
    
    for step in range(steps):
        idx = torch.randint(0, n_train, (batch_size,))
        xb = train_x[idx].to(device)
        yb = train_y[idx].to(device)
        
        loss = model(xb, yb)
        loss.backward()
        
        # Gradient scaling for recurrent model
        if hasattr(model, 'R'):
            for bank_list in [model.q_bank, model.k_bank, model.v_bank, 
                             model.out_bank, model.mlp_up_bank, model.mlp_down_bank]:
                for p in bank_list:
                    if p.grad is not None:
                        p.grad.mul_(1.0 / model.R)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        
        losses.append(loss.item())
        if step % 100 == 0:
            print(f"  Step {step:4d}: loss={loss.item():.4f}")
    
    elapsed = time.time() - t0
    print(f"  Training: {elapsed:.1f}s ({elapsed/steps*1000:.1f} ms/step)")
    
    # Validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for i in range(0, val_x.shape[0], batch_size):
            xb = val_x[i:i+batch_size].to(device)
            yb = val_y[i:i+batch_size].to(device)
            
            if hasattr(model, 'R'):
                loss = model(xb, yb, deep_supervision=False)
            else:
                loss = model(xb, yb)
            val_losses.append(loss.item())
    
    val_loss = sum(val_losses) / len(val_losses)
    # BPB = cross_entropy_loss / ln(2) (converts nats to bits)
    # But this is per-token. For per-byte, divide by avg bytes per token.
    # For character-level: 1 token ≈ 1 byte, so BPB ≈ loss / ln(2)
    bpb = val_loss / math.log(2)
    
    print(f"\n  Val Loss:  {val_loss:.4f} nats")
    print(f"  Val BPB:   {bpb:.4f} bits/byte")
    print(f"  Train Loss: {losses[0]:.4f} → {losses[-1]:.4f}")
    
    return {
        "name": name,
        "params": n_params,
        "val_loss": val_loss,
        "val_bpb": bpb,
        "train_loss_start": losses[0],
        "train_loss_end": losses[-1],
        "elapsed_sec": elapsed,
    }


# ============================================================
# MAIN: Head-to-head comparison
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  PARAMETER GOLF — LOCAL BPB ESTIMATION")
    print(f"  Device: {device}")
    print("=" * 60)
    
    SEQ_LEN = 256
    STEPS = 500
    BATCH = 16
    
    # Load data
    print("\n--- Loading data ---")
    result = get_text_data(seq_len=SEQ_LEN, num_seqs=3000)
    if result[0] is None:
        print("Using synthetic data")
        vocab_size = 128
        train_x = torch.randint(0, vocab_size, (2000, SEQ_LEN))
        train_y = torch.randint(0, vocab_size, (2000, SEQ_LEN))
        val_x = torch.randint(0, vocab_size, (500, SEQ_LEN))
        val_y = torch.randint(0, vocab_size, (500, SEQ_LEN))
    else:
        train_x, train_y, vocab_size = result
        # Split
        val_x = train_x[2000:]
        val_y = train_y[2000:]
        train_x = train_x[:2000]
        train_y = train_y[:2000]
    
    print(f"  Vocab: {vocab_size} | Train: {train_x.shape} | Val: {val_x.shape}")
    
    # ---- CONFIG A: Standard 9-layer (baseline-equivalent) ----
    # Match parameter count roughly to the competition baseline
    dim_a = 256  # Scaled down for local GPU
    layers_a = 9
    
    model_a = StandardTransformer(vocab_size, dim_a, layers_a, n_heads=8)
    result_a = train_and_evaluate(
        model_a, train_x, train_y, val_x, val_y,
        f"BASELINE: Standard {layers_a}L @ dim={dim_a}",
        steps=STEPS, batch_size=BATCH
    )
    del model_a
    torch.cuda.empty_cache() if device == 'cuda' else None
    
    # ---- CONFIG B: Depth-recurrent 4x3 (our fork, same effective depth) ----
    # 4 unique blocks × 3 reps = 12 effective layers
    # Wider dim since we save on params from weight sharing
    K, R = 4, 3
    dim_b = 320  # Can be wider because we share weights
    
    model_b = RecurrentTransformer(vocab_size, dim_b, K, R, n_heads=8, lora_rank=8)
    result_b = train_and_evaluate(
        model_b, train_x, train_y, val_x, val_y,
        f"OUR FORK: Recurrent {K}×{R} @ dim={dim_b} + LoRA-8 + Deep Supervision",
        steps=STEPS, batch_size=BATCH
    )
    del model_b
    torch.cuda.empty_cache() if device == 'cuda' else None
    
    # ---- CONFIG C: Depth-recurrent 6x2 palindromic ----
    K2, R2 = 6, 2
    dim_c = 288
    
    model_c = RecurrentTransformer(vocab_size, dim_c, K2, R2, n_heads=8, lora_rank=8)
    result_c = train_and_evaluate(
        model_c, train_x, train_y, val_x, val_y,
        f"OUR FORK: Palindromic {K2}×{R2} @ dim={dim_c} + LoRA-8",
        steps=STEPS, batch_size=BATCH
    )
    del model_c
    
    # ---- COMPARISON ----
    print("\n" + "=" * 60)
    print("  HEAD-TO-HEAD RESULTS")
    print("=" * 60)
    
    results = [result_a, result_b, result_c]
    for r in results:
        print(f"\n  {r['name']}")
        print(f"    Params:   {r['params']:,}")
        print(f"    Val BPB:  {r['val_bpb']:.4f}")
        print(f"    Val Loss: {r['val_loss']:.4f}")
        print(f"    Speed:    {r['elapsed_sec']/STEPS*1000:.1f} ms/step")
    
    # Delta
    baseline_bpb = result_a['val_bpb']
    fork1_bpb = result_b['val_bpb']
    fork2_bpb = result_c['val_bpb']
    
    delta1 = baseline_bpb - fork1_bpb
    delta2 = baseline_bpb - fork2_bpb
    
    print(f"\n  {'='*60}")
    print(f"  BPB Improvement (positive = our fork is better):")
    print(f"    4×3 fork: {delta1:+.4f} BPB ({delta1/baseline_bpb*100:+.2f}%)")
    print(f"    6×2 fork: {delta2:+.4f} BPB ({delta2/baseline_bpb*100:+.2f}%)")
    
    # Extrapolation
    print(f"\n  {'='*60}")
    print(f"  EXTRAPOLATION TO COMPETITION:")
    print(f"  Current #1: 1.1194 BPB")
    
    if delta1 > 0:
        est1 = 1.1194 - delta1 * 0.5  # Conservative 50% scaling factor
        print(f"  4×3 est:    ~{est1:.4f} BPB (conservative)")
    else:
        print(f"  4×3 est:    fork did NOT improve over baseline locally")
    
    if delta2 > 0:
        est2 = 1.1194 - delta2 * 0.5
        print(f"  6×2 est:    ~{est2:.4f} BPB (conservative)")
    else:
        print(f"  6×2 est:    fork did NOT improve over baseline locally")
    
    print(f"\n  NOTE: Local test uses char-level on Shakespeare (not FineWeb).")
    print(f"  Actual results on FineWeb with Parallel Muon + QAT + TTT")
    print(f"  will likely be different. This test shows RELATIVE improvement.")
    print(f"  {'='*60}")
