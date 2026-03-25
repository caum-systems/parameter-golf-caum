"""
Parameter Golf Architecture Sweep — Local Experiments
Runs multiple model configurations on RTX 4070 SUPER and compares
learning curves over 500 steps. Relative BPB improvement indicates
which architecture will perform best on 8xH100.

Key changes to test (from SOTA analysis):
1. More layers: 9 → 11 (SOTA uses 11)
2. MLP expansion: 2x → 3x (SOTA uses 3x)
3. Both together (11L + 3x MLP)
4. Larger vocab: 1024 → 2048 (better compression)
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
from train_gpt import GPT, Hyperparameters, quantize_state_dict_int8

# ---- Data loading (same as baseline) ----
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

# ---- Experiment configs ----
CONFIGS = {
    "baseline_9L_2x": {
        "num_layers": 9, "mlp_mult": 2, "model_dim": 512,
        "num_heads": 8, "num_kv_heads": 4,
    },
    "deeper_11L_2x": {
        "num_layers": 11, "mlp_mult": 2, "model_dim": 512,
        "num_heads": 8, "num_kv_heads": 4,
    },
    "wider_mlp_9L_3x": {
        "num_layers": 9, "mlp_mult": 3, "model_dim": 512,
        "num_heads": 8, "num_kv_heads": 4,
    },
    "sota_style_11L_3x": {
        "num_layers": 11, "mlp_mult": 3, "model_dim": 512,
        "num_heads": 8, "num_kv_heads": 4,
    },
    "compact_11L_3x_dim448": {
        "num_layers": 11, "mlp_mult": 3, "model_dim": 448,
        "num_heads": 8, "num_kv_heads": 4,  # head_dim = 56
    },
}

NUM_STEPS = 500
BATCH_TOKENS = 8192
EVAL_EVERY = 100

def run_experiment(name, config, device, train_tokens, val_tokens, sp, seq_len):
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {name}")
    print(f"  Config: {config}")
    print(f"{'='*60}")

    args = Hyperparameters()
    
    # Override config
    num_layers = config["num_layers"]
    mlp_mult = config["mlp_mult"]
    model_dim = config["model_dim"]
    num_heads = config["num_heads"]
    num_kv_heads = config["num_kv_heads"]
    
    # Check head_dim is valid
    head_dim = model_dim // num_heads
    if model_dim % num_heads != 0:
        print(f"  SKIP: model_dim {model_dim} not divisible by num_heads {num_heads}")
        return None
    if head_dim % 2 != 0:
        print(f"  SKIP: head_dim {head_dim} must be even for RoPE")
        return None

    # Build model
    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=num_layers,
        model_dim=model_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        mlp_mult=mlp_mult,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Check quantized size
    obj, stats = quantize_state_dict_int8(model.state_dict())
    buf = io.BytesIO()
    torch.save(obj, buf)
    compressed = zlib.compress(buf.getvalue(), 9)
    code_bytes = 50000  # approx train_gpt.py size
    artifact_size = len(compressed) + code_bytes
    print(f"  Estimated artifact: {artifact_size/1e6:.2f} MB {'✅' if artifact_size <= 16_000_000 else '❌ TOO BIG'}")
    
    if artifact_size > 16_000_000:
        print(f"  SKIP: artifact too large ({artifact_size/1e6:.2f} MB > 16 MB)")
        del model, obj
        torch.cuda.empty_cache()
        return None

    # BPB lookup
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    batch_seqs = BATCH_TOKENS // seq_len

    # Eval function
    def evaluate():
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        total_bytes = 0
        n_seqs = min((val_tokens.numel() - 1) // seq_len, 100)  # fast eval
        with torch.inference_mode():
            for i in range(0, n_seqs, batch_seqs):
                end = min(i + batch_seqs, n_seqs)
                start_tok = i * seq_len
                end_tok = end * seq_len + 1
                local = val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
                x = local[:-1].reshape(-1, seq_len)
                y = local[1:].reshape(-1, seq_len)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(x, y).detach()
                batch_count = float(y.numel())
                total_loss += loss.item() * batch_count
                total_tokens += batch_count
                prev_ids = x.reshape(-1)
                tgt_ids = y.reshape(-1)
                token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
                token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
                total_bytes += token_bytes.to(torch.float64).sum().item()
        avg_loss = total_loss / total_tokens
        bpb = (avg_loss / math.log(2.0)) * (total_tokens / total_bytes)
        model.train()
        return avg_loss, bpb

    # Training loop
    results = {
        "name": name,
        "config": config,
        "params": n_params,
        "artifact_mb": artifact_size / 1e6,
        "steps": [],
        "losses": [],
        "bpbs": [],
    }

    pos = 0
    t0 = time.time()
    
    for step in range(NUM_STEPS):
        # Get batch
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
            results["steps"].append(step + 1)
            results["losses"].append(val_loss)
            results["bpbs"].append(val_bpb)
            print(f"  step {step+1:>4}/{NUM_STEPS}: val_loss={val_loss:.4f} val_bpb={val_bpb:.4f} | {tps:.0f} tok/s | {elapsed:.1f}s")

    # Final eval
    val_loss, val_bpb = evaluate()
    results["final_loss"] = val_loss
    results["final_bpb"] = val_bpb
    results["total_time"] = time.time() - t0
    results["tok_per_sec"] = NUM_STEPS * BATCH_TOKENS / results["total_time"]

    print(f"\n  FINAL: val_loss={val_loss:.4f} val_bpb={val_bpb:.4f}")
    print(f"  Time: {results['total_time']:.1f}s | Speed: {results['tok_per_sec']:.0f} tok/s")

    # Cleanup
    del model, obj, optimizer
    torch.cuda.empty_cache()
    
    return results


def main():
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True

    args = Hyperparameters()
    seq_len = args.train_seq_len

    print("=" * 60)
    print("  PARAMETER GOLF — ARCHITECTURE SWEEP")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Steps per config: {NUM_STEPS}")
    print(f"  Batch: {BATCH_TOKENS} tokens")
    print(f"  Configs to test: {len(CONFIGS)}")
    print("=" * 60)

    # Load tokenizer
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)

    # Load data
    val_files = sorted(glob.glob(args.val_files))
    val_tokens = torch.cat([load_data_shard(Path(f)) for f in val_files]).contiguous()
    train_files = sorted(glob.glob(args.train_files))
    train_tokens = torch.cat([load_data_shard(Path(f)) for f in train_files]).contiguous()
    print(f"  Train: {train_tokens.numel():,} tokens | Val: {val_tokens.numel():,} tokens")

    # Run all experiments
    all_results = []
    for name, config in CONFIGS.items():
        try:
            result = run_experiment(name, config, device, train_tokens, val_tokens, sp, seq_len)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"  ERROR in {name}: {e}")
            torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Config':<25} {'Params':>10} {'Artifact':>10} {'Final BPB':>10} {'Speed':>10} {'vs Baseline':>12}")
    print("-" * 80)
    
    baseline_bpb = None
    for r in sorted(all_results, key=lambda x: x["final_bpb"]):
        if baseline_bpb is None and r["name"] == "baseline_9L_2x":
            baseline_bpb = r["final_bpb"]
        delta = ""
        if baseline_bpb and r["name"] != "baseline_9L_2x":
            diff = r["final_bpb"] - baseline_bpb
            delta = f"{diff:+.4f}"
        print(f"{r['name']:<25} {r['params']:>10,} {r['artifact_mb']:>9.2f}M {r['final_bpb']:>10.4f} {r['tok_per_sec']:>9.0f}/s {delta:>12}")

    # Save results
    results_path = Path("experiments/sweep_results.json")
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Winner
    if all_results:
        best = min(all_results, key=lambda x: x["final_bpb"])
        print(f"\n🏆 BEST: {best['name']} with BPB={best['final_bpb']:.4f}")


if __name__ == "__main__":
    main()
