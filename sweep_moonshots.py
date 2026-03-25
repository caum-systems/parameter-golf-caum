"""
Parameter Golf Round 3 — MOONSHOT EXPERIMENTS
=============================================
These are novel architectures that NO current submission uses.
The goal is to find a fundamentally better approach, not incremental gains.

IDEA 1: DEPTH RECURRENCE (Weight Sharing)
- Instead of 11 UNIQUE layers (11 sets of weights), use 4 UNIQUE layers
  applied 3x each (= 12 effective layers, but only 4 sets of weights)
- This cuts model size by ~3x → we can make each layer MUCH wider
- The model effectively has 12 layers of depth but only 4 layers of weight budget
- Savings go into wider dims → better representational capacity per byte

IDEA 2: PROGRESSIVE DEPTH (Start shallow, grow deep during training)
- Train a 4-layer model for the first 40% of training (fast convergence)
- Then "unfold" shared weights into more layers for the last 60%
- Model learns basic patterns fast, then refines with more depth

IDEA 3: HYBRID RECURRENT + ATTENTION
- Replace some attention layers with a simple linear recurrence (RWKV-style)
- Recurrent layers are MUCH cheaper (no O(n²) attention)
- This saves compute → can train more steps in 10 minutes
"""
import os, sys, time, math, glob, json, io, zlib, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import sentencepiece as spm

import torch.distributed
torch.distributed.is_available = lambda: False
torch.distributed.is_initialized = lambda: False

sys.path.insert(0, '.')
from train_gpt import (
    Hyperparameters, RMSNorm, CastedLinear, Rotary, apply_rotary_emb,
    quantize_state_dict_int8
)

def load_data_shard(file):
    header = np.fromfile(file, dtype="<i4", count=256)
    num_tokens = int(header[2])
    offset = 256 * np.dtype("<i4").itemsize
    return torch.from_numpy(np.fromfile(file, dtype="<u2", count=num_tokens, offset=offset).astype(np.uint16, copy=False))

def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


# ============================================================
# MOONSHOT 1: DEPTH RECURRENCE GPT
# ============================================================

class BigramHash(nn.Module):
    def __init__(self, num_buckets, embed_dim):
        super().__init__()
        self.num_buckets = num_buckets
        self.embedding = nn.Embedding(num_buckets, embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.01)
        self.register_buffer('hash_a', torch.tensor(2654435761, dtype=torch.long))
        self.register_buffer('hash_b', torch.tensor(40503, dtype=torch.long))

    def forward(self, input_ids):
        prev_ids = torch.zeros_like(input_ids)
        prev_ids[:, 1:] = input_ids[:, :-1]
        bigram_hash = (self.hash_a * prev_ids.long() + self.hash_b * input_ids.long()) % self.num_buckets
        return self.embedding(bigram_hash)


class RecurrentBlock(nn.Module):
    """A transformer block that can be applied multiple times (weight sharing)."""
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        head_dim = dim // num_heads
        kv_dim = num_kv_heads * head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.attn_proj = CastedLinear(dim, dim, bias=False)
        self.attn_proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(head_dim, base=rope_base)
        # MLP with LeakyReLU²
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.mlp_proj = CastedLinear(hidden, dim, bias=False)
        self.mlp_proj._zero_init = True
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x, pass_idx=0):
        """pass_idx: which recurrence pass (for potential layer-specific behavior)"""
        # Attention
        bsz, seqlen, dim = x.shape
        normed = self.attn_norm(x)
        q = self.c_q(normed).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(normed).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(normed).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           enable_gqa=(self.num_kv_heads != self.num_heads))
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn_proj(y)
        # MLP
        mlp_in = F.leaky_relu(self.fc(self.mlp_norm(x)), negative_slope=0.5)
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp_proj(mlp_in.square())
        return x


class DepthRecurrentGPT(nn.Module):
    """
    MOONSHOT: Depth Recurrence
    Instead of N unique layers, use K unique blocks applied R times each.
    Total effective depth = K * R, but only K blocks worth of parameters.
    """
    def __init__(self, vocab_size, num_unique_blocks, recurrence_factor, model_dim,
                 num_heads, num_kv_heads, mlp_mult, logit_softcap=30.0,
                 rope_base=10000.0, qk_gain_init=1.5, bigram_buckets=1536, bigram_dim=128):
        super().__init__()
        self.logit_softcap = logit_softcap
        self.recurrence_factor = recurrence_factor
        self.num_unique_blocks = num_unique_blocks
        effective_depth = num_unique_blocks * recurrence_factor
        
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.005)
        
        self.bigram_hash = BigramHash(bigram_buckets, bigram_dim)
        self.bigram_proj = CastedLinear(bigram_dim, model_dim, bias=False)
        nn.init.zeros_(self.bigram_proj.weight)
        
        # Only K unique blocks (but applied R times each)
        self.blocks = nn.ModuleList([
            RecurrentBlock(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for _ in range(num_unique_blocks)
        ])
        
        # Per-pass scaling (different scale for each recurrence pass)
        self.pass_scales = nn.ParameterList([
            nn.Parameter(torch.ones(model_dim, dtype=torch.float32) / math.sqrt(p + 1))
            for p in range(recurrence_factor)
        ])
        
        self.final_norm = RMSNorm()
        
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward(self, input_ids, target_ids):
        x = self.tok_emb(input_ids)
        x = x + self.bigram_proj(self.bigram_hash(input_ids))
        x = F.rms_norm(x, (x.size(-1),))
        
        # Apply each block R times
        for pass_idx in range(self.recurrence_factor):
            scale = self.pass_scales[pass_idx].to(dtype=x.dtype)[None, None, :]
            for block in self.blocks:
                x = block(x * scale, pass_idx)
        
        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        logits = F.linear(x, self.tok_emb.weight)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


# ============================================================
# MOONSHOT 2: WIDE RECURRENT (fewer blocks, wider dim)
# ============================================================

class WideRecurrentGPT(DepthRecurrentGPT):
    """Same as DepthRecurrent but uses the parameter savings for a wider model."""
    pass  # Same architecture, just different hyperparameters (wider dim)


# ============================================================
# MOONSHOT 3: HYBRID LINEAR RECURRENCE + ATTENTION
# ============================================================

class SimpleLinearRecurrence(nn.Module):
    """RWKV-inspired linear recurrence. O(n) instead of O(n²) attention.
    Much faster, allowing more training steps in 10 minutes."""
    def __init__(self, dim):
        super().__init__()
        self.key_proj = CastedLinear(dim, dim, bias=False)
        self.value_proj = CastedLinear(dim, dim, bias=False)
        self.receptance_proj = CastedLinear(dim, dim, bias=False)
        self.output_proj = CastedLinear(dim, dim, bias=False)
        self.output_proj._zero_init = True
        self.time_decay = nn.Parameter(torch.randn(dim) * 0.01 - 5.0)  # learned decay
        self.time_first = nn.Parameter(torch.randn(dim) * 0.01)  # bonus for current
        
    def forward(self, x):
        bsz, seqlen, dim = x.shape
        k = self.key_proj(x)  # (B, T, D)
        v = self.value_proj(x)  # (B, T, D)
        r = torch.sigmoid(self.receptance_proj(x))  # (B, T, D) gate
        
        # Time decay
        w = -torch.exp(self.time_decay)  # force negative = decay
        u = self.time_first
        
        # Sequential scan (simplified RWKV)
        outputs = []
        state = torch.zeros(bsz, dim, device=x.device, dtype=x.dtype)
        for t in range(seqlen):
            kt, vt = k[:, t], v[:, t]
            # WKV attention analog
            wkv = state + torch.exp(u) * kt * vt
            state = torch.exp(w) * state + kt * vt
            outputs.append(r[:, t] * wkv)
        
        output = torch.stack(outputs, dim=1)
        return self.output_proj(output)


class HybridBlock(nn.Module):
    """Block that uses either attention or linear recurrence."""
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                 use_recurrence=False):
        super().__init__()
        self.norm1 = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.use_recurrence = use_recurrence
        
        if use_recurrence:
            self.mixer = SimpleLinearRecurrence(dim)
        else:
            # Standard attention
            head_dim = dim // num_heads
            kv_dim = num_kv_heads * head_dim
            self.num_heads = num_heads
            self.num_kv_heads = num_kv_heads
            self.head_dim = head_dim
            self.c_q = CastedLinear(dim, dim, bias=False)
            self.c_k = CastedLinear(dim, kv_dim, bias=False)
            self.c_v = CastedLinear(dim, kv_dim, bias=False)
            self.attn_proj = CastedLinear(dim, dim, bias=False)
            self.attn_proj._zero_init = True
            self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
            self.rotary = Rotary(head_dim, base=rope_base)
        
        self.mixer_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.mlp_proj = CastedLinear(hidden, dim, bias=False)
        self.mlp_proj._zero_init = True
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x):
        normed = self.norm1(x)
        if self.use_recurrence:
            mixer_out = self.mixer(normed)
        else:
            bsz, seqlen, dim = normed.shape
            q = self.c_q(normed).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.c_k(normed).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = self.c_v(normed).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
            q = F.rms_norm(q, (q.size(-1),))
            k = F.rms_norm(k, (k.size(-1),))
            cos, sin = self.rotary(seqlen, x.device, q.dtype)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
            q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                               enable_gqa=(self.num_kv_heads != self.num_heads))
            mixer_out = self.attn_proj(y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim))
        
        x = x + self.mixer_scale.to(dtype=x.dtype)[None, None, :] * mixer_out
        mlp_in = F.leaky_relu(self.fc(self.mlp_norm(x)), negative_slope=0.5)
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp_proj(mlp_in.square())
        return x


class HybridGPT(nn.Module):
    """Mix attention + linear recurrence layers."""
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads, mlp_mult,
                 recurrence_layers=None, logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
                 bigram_buckets=1536, bigram_dim=128):
        super().__init__()
        self.logit_softcap = logit_softcap
        if recurrence_layers is None:
            recurrence_layers = set()
        
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.005)
        self.bigram_hash = BigramHash(bigram_buckets, bigram_dim)
        self.bigram_proj = CastedLinear(bigram_dim, model_dim, bias=False)
        nn.init.zeros_(self.bigram_proj.weight)
        
        self.blocks = nn.ModuleList([
            HybridBlock(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                       use_recurrence=(i in recurrence_layers))
            for i in range(num_layers)
        ])
        self.final_norm = RMSNorm()
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward(self, input_ids, target_ids):
        x = self.tok_emb(input_ids)
        x = x + self.bigram_proj(self.bigram_hash(input_ids))
        x = F.rms_norm(x, (x.size(-1),))
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        logits = F.linear(x, self.tok_emb.weight)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

NUM_STEPS = 500
BATCH_TOKENS = 8192
EVAL_EVERY = 100


def build_model(name, args):
    if name == "best_so_far":
        # Our current best: 11L/3x + LeakyReLU² + BigramHash (from Round 2)
        from sweep_enhancements import EnhancedGPT
        return EnhancedGPT(
            vocab_size=args.vocab_size, num_layers=11, model_dim=512,
            num_heads=8, num_kv_heads=4, mlp_mult=3,
            use_bigram_hash=True, bigram_buckets=1536, bigram_dim=128,
            use_smeargate=False,
        )
    
    elif name == "recurrent_4x3_512":
        # 4 unique blocks × 3 passes = 12 effective depth, but only 4 blocks of params
        return DepthRecurrentGPT(
            vocab_size=args.vocab_size, num_unique_blocks=4, recurrence_factor=3,
            model_dim=512, num_heads=8, num_kv_heads=4, mlp_mult=3,
        )
    
    elif name == "recurrent_4x3_768":
        # MOONSHOT: Use param savings from recurrence → go MUCH wider (768 dim)
        return DepthRecurrentGPT(
            vocab_size=args.vocab_size, num_unique_blocks=4, recurrence_factor=3,
            model_dim=768, num_heads=12, num_kv_heads=4, mlp_mult=3,
        )
    
    elif name == "recurrent_6x2_640":
        # 6 unique blocks × 2 passes = 12 effective depth, wider at 640
        return DepthRecurrentGPT(
            vocab_size=args.vocab_size, num_unique_blocks=6, recurrence_factor=2,
            model_dim=640, num_heads=8, num_kv_heads=4, mlp_mult=3,
        )
    
    elif name == "hybrid_rwkv_attn":
        # Replace layers 0,1,2 with linear recurrence (fast local), keep 3-10 as attention
        return HybridGPT(
            vocab_size=args.vocab_size, num_layers=11, model_dim=512,
            num_heads=8, num_kv_heads=4, mlp_mult=3,
            recurrence_layers={0, 1, 2},
        )
    
    else:
        raise ValueError(f"Unknown model: {name}")


def run_experiment(name, device, train_tokens, val_tokens, sp, seq_len):
    print(f"\n{'='*60}")
    print(f"  🚀 MOONSHOT: {name}")
    print(f"{'='*60}")

    args = Hyperparameters()
    
    try:
        model = build_model(name, args).to(device)
    except Exception as e:
        print(f"  BUILD ERROR: {e}")
        return None

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    obj, stats = quantize_state_dict_int8(model.state_dict())
    buf = io.BytesIO()
    torch.save(obj, buf)
    compressed = zlib.compress(buf.getvalue(), 9)
    artifact_size = len(compressed) + 50000
    print(f"  Artifact: {artifact_size/1e6:.2f} MB {'✅' if artifact_size <= 16_000_000 else '❌ TOO BIG'}")
    del obj
    
    if artifact_size > 16_000_000:
        print(f"  SKIP: artifact too large")
        del model
        torch.cuda.empty_cache()
        return None

    base_bytes_lut, has_space_lut, is_boundary_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    batch_seqs = BATCH_TOKENS // seq_len

    def evaluate():
        model.eval()
        total_loss = total_tokens = total_bytes = 0
        n_seqs = min((val_tokens.numel() - 1) // seq_len, 100)
        with torch.inference_mode():
            for i in range(0, n_seqs, batch_seqs):
                end = min(i + batch_seqs, n_seqs)
                start_tok, end_tok = i * seq_len, end * seq_len + 1
                local = val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
                x = local[:-1].reshape(-1, seq_len)
                y = local[1:].reshape(-1, seq_len)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(x, y).detach()
                count = float(y.numel())
                total_loss += loss.item() * count
                total_tokens += count
                prev_ids, tgt_ids = x.reshape(-1), y.reshape(-1)
                tb = base_bytes_lut[tgt_ids].to(torch.int16)
                tb += (has_space_lut[tgt_ids] & ~is_boundary_lut[prev_ids]).to(torch.int16)
                total_bytes += tb.to(torch.float64).sum().item()
        avg_loss = total_loss / total_tokens
        bpb = (avg_loss / math.log(2.0)) * (total_tokens / total_bytes)
        model.train()
        return avg_loss, bpb

    pos = 0
    t0 = time.time()
    for step in range(NUM_STEPS):
        need = batch_seqs * seq_len + 1
        if pos + need > train_tokens.numel():
            pos = 0
        chunk = train_tokens[pos:pos + need].to(device=device, dtype=torch.int64)
        pos += need
        x = chunk[:-1].reshape(-1, seq_len)
        y = chunk[1:].reshape(-1, seq_len)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % EVAL_EVERY == 0:
            val_loss, val_bpb = evaluate()
            elapsed = time.time() - t0
            tps = (step + 1) * BATCH_TOKENS / elapsed
            print(f"  step {step+1:>4}: bpb={val_bpb:.4f} loss={val_loss:.4f} | {tps:.0f} tok/s")

    val_loss, val_bpb = evaluate()
    total_time = time.time() - t0
    print(f"  FINAL: bpb={val_bpb:.4f} | {total_time:.1f}s | {NUM_STEPS*BATCH_TOKENS/total_time:.0f} tok/s")

    result = {
        "name": name, "params": n_params, "artifact_mb": artifact_size/1e6,
        "final_bpb": val_bpb, "final_loss": val_loss,
        "tok_per_sec": NUM_STEPS * BATCH_TOKENS / total_time,
    }
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def main():
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True

    args = Hyperparameters()
    seq_len = args.train_seq_len
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)

    print("=" * 60)
    print("  🚀 PARAMETER GOLF — MOONSHOT EXPERIMENTS")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    val_tokens = torch.cat([load_data_shard(Path(f)) for f in sorted(glob.glob(args.val_files))]).contiguous()
    train_tokens = torch.cat([load_data_shard(Path(f)) for f in sorted(glob.glob(args.train_files))]).contiguous()
    print(f"  Train: {train_tokens.numel():,} | Val: {val_tokens.numel():,}")

    MODELS = [
        "best_so_far",           # baseline comparison
        "recurrent_4x3_512",     # 4 blocks × 3 passes, 512dim
        "recurrent_4x3_768",     # 4 blocks × 3 passes, 768dim (WIDE)
        "recurrent_6x2_640",     # 6 blocks × 2 passes, 640dim
        "hybrid_rwkv_attn",      # 3 RWKV + 8 attention layers
    ]

    all_results = []
    for name in MODELS:
        try:
            result = run_experiment(name, device, train_tokens, val_tokens, sp, seq_len)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"  ERROR in {name}: {e}")
            import traceback; traceback.print_exc()
            torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("  🚀 MOONSHOT RESULTS")
    print("=" * 80)
    print(f"{'Config':<25} {'Params':>10} {'Artifact':>10} {'BPB':>10} {'Speed':>10}")
    print("-" * 70)
    best_bpb = None
    for r in sorted(all_results, key=lambda x: x["final_bpb"]):
        if best_bpb is None:
            best_bpb = r["final_bpb"]
        delta = f" ({r['final_bpb']-all_results[0]['final_bpb']:+.4f})" if r != all_results[0] else " (baseline)"
        print(f"{r['name']:<25} {r['params']:>10,} {r['artifact_mb']:>9.2f}M {r['final_bpb']:>9.4f} {r['tok_per_sec']:>9.0f}/s{delta}")

    results_path = Path("experiments/moonshot_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    if all_results:
        best = min(all_results, key=lambda x: x["final_bpb"])
        print(f"\n🏆 BEST: {best['name']} — BPB={best['final_bpb']:.4f} ({best['artifact_mb']:.2f}MB)")


if __name__ == "__main__":
    main()
