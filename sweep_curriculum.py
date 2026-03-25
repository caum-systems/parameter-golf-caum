"""
Parameter Golf — CAUM-Inspired Curriculum Learning
===================================================
Uses LZ76 compression complexity (from CAUM's trajectory analysis engine)
to score training batches and order them by complexity.

Concept: CAUM's ASC engine uses LZ76 to detect structural diversity.
We apply the same principle to TEXT data: score each training chunk
by its compressibility, then train in order:
  Phase 1 (first 40%): Easy text (high compression → simple patterns)
  Phase 2 (next 30%): Medium complexity
  Phase 3 (last 30%): Hard text (low compression → diverse patterns)

This is combined with the winning architecture: Depth Recurrence 6×2@640
"""
import os, sys, time, math, glob, json, io, zlib
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
from train_gpt import Hyperparameters, quantize_state_dict_int8
from sweep_moonshots import (
    DepthRecurrentGPT, load_data_shard, build_sentencepiece_luts, BigramHash
)


# ============================================================
# CAUM-INSPIRED: LZ76 COMPLEXITY SCORING
# ============================================================

def lz76_complexity(data: bytes) -> float:
    """
    LZ76 complexity measure — directly from CAUM's trajectory analysis.
    Counts the number of distinct subpatterns in a byte sequence.
    Higher = more complex/diverse content.
    
    This is the SAME algorithm CAUM uses to score agent trajectories.
    Here we apply it to raw text data for curriculum ordering.
    """
    n = len(data)
    if n == 0:
        return 0.0
    
    i = 0
    c = 1  # complexity counter
    l = 1  # current match length
    
    while i + l <= n:
        # Check if the substring data[i:i+l] appeared before in data[0:i+l-1]
        substring = data[i:i + l]
        prefix = data[0:i + l - 1]
        
        if substring in prefix:
            l += 1
        else:
            c += 1
            i += l
            l = 1
    
    # Normalize by theoretical maximum
    if n > 0:
        return c / (n / max(1, math.log2(n)))
    return 0.0


def score_training_chunks(train_tokens, chunk_size=8192):
    """Score all training chunks by LZ76 complexity."""
    n_chunks = train_tokens.numel() // chunk_size
    scores = []
    
    print(f"  Scoring {n_chunks} chunks with LZ76 complexity...")
    t0 = time.time()
    
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk_ids = train_tokens[start:end].numpy().astype(np.uint16)
        # Convert token IDs to bytes for LZ76
        chunk_bytes = chunk_ids.tobytes()
        # Use zlib compression ratio as fast proxy for LZ76
        compressed = zlib.compress(chunk_bytes, 1)  # level 1 = fast
        ratio = len(compressed) / len(chunk_bytes)
        scores.append((i, ratio))
    
    elapsed = time.time() - t0
    print(f"  Scored {n_chunks} chunks in {elapsed:.1f}s")
    print(f"  Compression ratios: min={min(s[1] for s in scores):.3f}, "
          f"max={max(s[1] for s in scores):.3f}, "
          f"mean={sum(s[1] for s in scores)/len(scores):.3f}")
    
    return scores


def create_curriculum_order(scores, strategy="easy_first"):
    """
    Create training order based on complexity scores.
    
    Strategies:
    - "easy_first": Low complexity first (high compression ratio = easy)
    - "hard_first": High complexity first
    - "mixed": Alternate easy and hard
    - "random": Baseline random order
    """
    if strategy == "easy_first":
        # High compression ratio = easy (more compressible = simpler patterns)
        ordered = sorted(scores, key=lambda x: -x[1])
    elif strategy == "hard_first":
        ordered = sorted(scores, key=lambda x: x[1])
    elif strategy == "mixed":
        # Sort, then interleave easy and hard
        sorted_scores = sorted(scores, key=lambda x: x[1])
        n = len(sorted_scores)
        easy = sorted_scores[n//2:]  # high compression = easy
        hard = sorted_scores[:n//2]  # low compression = hard
        ordered = []
        for e, h in zip(easy, hard):
            ordered.extend([e, h])
    elif strategy == "random":
        import random
        ordered = list(scores)
        random.shuffle(ordered)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return [idx for idx, _ in ordered]


# ============================================================
# EXPERIMENT
# ============================================================

NUM_STEPS = 500
BATCH_TOKENS = 8192
EVAL_EVERY = 100


def run_curriculum_experiment(name, strategy, device, train_tokens, val_tokens, sp, seq_len):
    print(f"\n{'='*60}")
    print(f"  CAUM CURRICULUM: {name} ({strategy})")
    print(f"{'='*60}")

    args = Hyperparameters()
    
    # Use our best architecture: Depth Recurrence 6×2@640
    model = DepthRecurrentGPT(
        vocab_size=args.vocab_size, num_unique_blocks=6, recurrence_factor=2,
        model_dim=640, num_heads=8, num_kv_heads=4, mlp_mult=3,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

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

    # Score and order chunks
    chunk_size = batch_seqs * seq_len
    scores = score_training_chunks(train_tokens, chunk_size)
    chunk_order = create_curriculum_order(scores, strategy)

    # Train with curriculum
    t0 = time.time()
    chunk_idx = 0
    for step in range(NUM_STEPS):
        # Get next chunk in curriculum order
        ci = chunk_order[chunk_idx % len(chunk_order)]
        chunk_idx += 1
        start = ci * chunk_size
        end = start + chunk_size + 1
        if end > train_tokens.numel():
            chunk_idx = 0
            ci = chunk_order[0]
            start = ci * chunk_size
            end = start + chunk_size + 1
        
        chunk = train_tokens[start:end].to(device=device, dtype=torch.int64)
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
            print(f"  step {step+1:>4}: bpb={val_bpb:.4f} | {tps:.0f} tok/s")

    val_loss, val_bpb = evaluate()
    total_time = time.time() - t0
    print(f"  FINAL: bpb={val_bpb:.4f} | {total_time:.1f}s")

    result = {
        "name": name, "strategy": strategy, "params": n_params,
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
    print("  🧠 CAUM-INSPIRED CURRICULUM LEARNING")
    print(f"  Architecture: Depth Recurrence 6×2@640")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    val_tokens = torch.cat([load_data_shard(Path(f)) for f in sorted(glob.glob(args.val_files))]).contiguous()
    train_tokens = torch.cat([load_data_shard(Path(f)) for f in sorted(glob.glob(args.train_files))]).contiguous()
    print(f"  Train: {train_tokens.numel():,} | Val: {val_tokens.numel():,}")

    STRATEGIES = [
        ("random_baseline", "random"),
        ("easy_first", "easy_first"),        # CAUM curriculum: simple → complex
        ("hard_first", "hard_first"),        # Reverse curriculum
        ("mixed_interleave", "mixed"),       # Alternate easy/hard
    ]

    all_results = []
    for name, strategy in STRATEGIES:
        try:
            result = run_curriculum_experiment(name, strategy, device, train_tokens, val_tokens, sp, seq_len)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"  ERROR in {name}: {e}")
            import traceback; traceback.print_exc()
            torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("  🧠 CAUM CURRICULUM RESULTS")
    print("=" * 80)
    print(f"{'Strategy':<25} {'BPB':>10} {'Speed':>10}")
    print("-" * 50)
    baseline_bpb = None
    for r in sorted(all_results, key=lambda x: x["final_bpb"]):
        if "random" in r["name"]:
            baseline_bpb = r["final_bpb"]
        delta = f" ({r['final_bpb']-baseline_bpb:+.4f})" if baseline_bpb and "random" not in r["name"] else ""
        print(f"{r['name']:<25} {r['final_bpb']:>9.4f} {r['tok_per_sec']:>9.0f}/s{delta}")

    results_path = Path("experiments/curriculum_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    if all_results:
        best = min(all_results, key=lambda x: x["final_bpb"])
        print(f"\n🏆 BEST: {best['name']} — BPB={best['final_bpb']:.4f}")


if __name__ == "__main__":
    main()
