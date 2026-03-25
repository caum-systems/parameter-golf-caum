"""
Parameter Golf Round 2 — Enhanced Architecture Test
Test the key improvements from top leaderboard entries:
1. LeakyReLU(0.5)² activation (−0.003 BPB from SOTA)
2. BigramHash embedding (character bigram features)
3. SmearGate (gate between current and previous token embeddings)
All tested on RTX 4070 SUPER with 500 steps, compared to baseline.
"""
import os, sys, time, math, glob, json, io, zlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import sentencepiece as spm

# Disable distributed
import torch.distributed
torch.distributed.is_available = lambda: False
torch.distributed.is_initialized = lambda: False

sys.path.insert(0, '.')
from train_gpt import (
    Hyperparameters, RMSNorm, CastedLinear, Rotary, apply_rotary_emb,
    quantize_state_dict_int8
)

# ---- Data loading ----
def load_data_shard(file):
    header = np.fromfile(file, dtype="<i4", count=256)
    num_tokens = int(header[2])
    offset = 256 * np.dtype("<i4").itemsize
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=offset)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

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
# ENHANCED ARCHITECTURE MODULES
# ============================================================

class BigramHash(nn.Module):
    """Hash consecutive token pairs into a learned embedding.
    Gives the model access to character-level bigram features cheaply."""
    def __init__(self, vocab_size: int, num_buckets: int, embed_dim: int):
        super().__init__()
        self.num_buckets = num_buckets
        self.embedding = nn.Embedding(num_buckets, embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.01)
        # Precompute hash coefficients
        self.register_buffer('hash_a', torch.tensor(2654435761, dtype=torch.long))  # golden ratio prime
        self.register_buffer('hash_b', torch.tensor(40503, dtype=torch.long))

    def forward(self, input_ids: Tensor) -> Tensor:
        """input_ids: (batch, seq_len)"""
        bsz, seq_len = input_ids.shape
        # Shift to get previous token (use 0 for first position)
        prev_ids = torch.zeros_like(input_ids)
        prev_ids[:, 1:] = input_ids[:, :-1]
        # Hash bigram: h = (a * prev + b * curr) % num_buckets
        bigram_hash = (
            self.hash_a * prev_ids.long() + self.hash_b * input_ids.long()
        ) % self.num_buckets
        return self.embedding(bigram_hash)


class SmearGate(nn.Module):
    """Gate between current and previous token representations.
    Helps the model attend to local context without attention."""
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(2 * dim, dim, bias=False)
        nn.init.zeros_(self.gate.weight)

    def forward(self, x: Tensor) -> Tensor:
        """x: (batch, seq_len, dim)"""
        # Shift: previous token representation
        x_prev = torch.zeros_like(x)
        x_prev[:, 1:] = x[:, :-1]
        gate_input = torch.cat([x, x_prev], dim=-1)
        gate_value = torch.sigmoid(self.gate(gate_input))
        return x + gate_value * (x_prev - x)


class EnhancedMLP(nn.Module):
    """MLP with LeakyReLU(0.5)² activation — from SOTA submission."""
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = F.leaky_relu(self.fc(x), negative_slope=0.5)
        return self.proj(x.square())


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           enable_gqa=(self.num_kv_heads != self.num_heads))
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class EnhancedBlock(nn.Module):
    """Block with enhanced MLP (LeakyReLU²) + optional SmearGate."""
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                 use_smeargate=True):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = EnhancedMLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.smeargate = SmearGate(dim) if use_smeargate else None

    def forward(self, x, x0):
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        if self.smeargate is not None:
            x = self.smeargate(x)
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class EnhancedGPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads,
                 mlp_mult, logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
                 use_bigram_hash=True, bigram_buckets=1536, bigram_dim=128,
                 use_smeargate=True, ln_scale=True):
        super().__init__()
        self.logit_softcap = logit_softcap
        self.ln_scale = ln_scale
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.005)

        # BigramHash
        self.bigram_hash = BigramHash(vocab_size, bigram_buckets, bigram_dim) if use_bigram_hash else None
        if use_bigram_hash:
            self.bigram_proj = CastedLinear(bigram_dim, model_dim, bias=False)
            nn.init.zeros_(self.bigram_proj.weight)

        # U-Net structure
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))

        # LN scale factors: 1/sqrt(layer_idx+1)
        self.blocks = nn.ModuleList([
            EnhancedBlock(model_dim, num_heads, num_kv_heads, mlp_mult,
                         rope_base, qk_gain_init, use_smeargate=use_smeargate)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm()

        # Zero-init output projections
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward(self, input_ids, target_ids):
        x = self.tok_emb(input_ids)
        if self.bigram_hash is not None:
            x = x + self.bigram_proj(self.bigram_hash(input_ids))
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []

        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        logits_proj = F.linear(x, self.tok_emb.weight)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

CONFIGS = {
    "baseline_11L_3x": {
        "num_layers": 11, "mlp_mult": 3, "model_dim": 512,
        "num_heads": 8, "num_kv_heads": 4,
        "use_bigram_hash": False, "use_smeargate": False,
    },
    "leaky_relu_only": {
        "num_layers": 11, "mlp_mult": 3, "model_dim": 512,
        "num_heads": 8, "num_kv_heads": 4,
        "use_bigram_hash": False, "use_smeargate": False,
        # LeakyReLU is always on in EnhancedGPT
    },
    "leaky_plus_smeargate": {
        "num_layers": 11, "mlp_mult": 3, "model_dim": 512,
        "num_heads": 8, "num_kv_heads": 4,
        "use_bigram_hash": False, "use_smeargate": True,
    },
    "leaky_plus_bigram": {
        "num_layers": 11, "mlp_mult": 3, "model_dim": 512,
        "num_heads": 8, "num_kv_heads": 4,
        "use_bigram_hash": True, "bigram_buckets": 1536, "bigram_dim": 128,
        "use_smeargate": False,
    },
    "full_enhanced": {
        "num_layers": 11, "mlp_mult": 3, "model_dim": 512,
        "num_heads": 8, "num_kv_heads": 4,
        "use_bigram_hash": True, "bigram_buckets": 1536, "bigram_dim": 128,
        "use_smeargate": True,
    },
}

NUM_STEPS = 500
BATCH_TOKENS = 8192
EVAL_EVERY = 100


def run_experiment(name, config, device, train_tokens, val_tokens, sp, seq_len):
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {name}")
    print(f"{'='*60}")

    args = Hyperparameters()
    
    model = EnhancedGPT(
        vocab_size=args.vocab_size,
        num_layers=config["num_layers"],
        model_dim=config["model_dim"],
        num_heads=config["num_heads"],
        num_kv_heads=config["num_kv_heads"],
        mlp_mult=config["mlp_mult"],
        use_bigram_hash=config.get("use_bigram_hash", False),
        bigram_buckets=config.get("bigram_buckets", 1536),
        bigram_dim=config.get("bigram_dim", 128),
        use_smeargate=config.get("use_smeargate", False),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    # Check artifact size
    obj, stats = quantize_state_dict_int8(model.state_dict())
    buf = io.BytesIO()
    torch.save(obj, buf)
    compressed = zlib.compress(buf.getvalue(), 9)
    artifact_size = len(compressed) + 50000
    print(f"  Artifact: {artifact_size/1e6:.2f} MB {'✅' if artifact_size <= 16_000_000 else '❌ TOO BIG'}")
    del obj
    
    if artifact_size > 16_000_000:
        print(f"  SKIP: too large")
        del model
        torch.cuda.empty_cache()
        return None

    # BPB lookup
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

    # Train
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
            print(f"  step {step+1:>4}: val_loss={val_loss:.4f} bpb={val_bpb:.4f} | {tps:.0f} tok/s")

    val_loss, val_bpb = evaluate()
    total_time = time.time() - t0
    print(f"  FINAL: bpb={val_bpb:.4f} | {total_time:.1f}s")

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
    print("  PARAMETER GOLF — ENHANCEMENT SWEEP")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Configs: {len(CONFIGS)}")
    print("=" * 60)

    val_tokens = torch.cat([load_data_shard(Path(f)) for f in sorted(glob.glob(args.val_files))]).contiguous()
    train_tokens = torch.cat([load_data_shard(Path(f)) for f in sorted(glob.glob(args.train_files))]).contiguous()
    print(f"  Train: {train_tokens.numel():,} | Val: {val_tokens.numel():,}")

    all_results = []
    for name, config in CONFIGS.items():
        try:
            result = run_experiment(name, config, device, train_tokens, val_tokens, sp, seq_len)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"  ERROR in {name}: {e}")
            import traceback; traceback.print_exc()
            torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 80)
    print("  RESULTS — ENHANCEMENT SWEEP")
    print("=" * 80)
    print(f"{'Config':<25} {'Params':>10} {'Artifact':>10} {'BPB':>10} {'Speed':>10}")
    print("-" * 70)
    baseline_bpb = None
    for r in sorted(all_results, key=lambda x: x["final_bpb"]):
        if "baseline" in r["name"]:
            baseline_bpb = r["final_bpb"]
        delta = f" ({r['final_bpb']-baseline_bpb:+.4f})" if baseline_bpb and "baseline" not in r["name"] else ""
        print(f"{r['name']:<25} {r['params']:>10,} {r['artifact_mb']:>9.2f}M {r['final_bpb']:>9.4f} {r['tok_per_sec']:>9.0f}/s{delta}")

    results_path = Path("experiments/enhancement_results.json")
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    if all_results:
        best = min(all_results, key=lambda x: x["final_bpb"])
        print(f"\n🏆 BEST: {best['name']} — BPB={best['final_bpb']:.4f}")


if __name__ == "__main__":
    main()
