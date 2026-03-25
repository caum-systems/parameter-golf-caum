"""
Single-GPU smoke test for Parameter Golf on Windows.
Loads the baseline model, runs a few training steps, and evaluates.
RTX 4070 SUPER (12GB) friendly.
"""
import os, sys, time, math, glob, zlib, io
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import sentencepiece as spm

# Monkey-patch: we don't use distributed
class FakeDist:
    @staticmethod
    def is_available(): return False
    @staticmethod
    def is_initialized(): return False

import torch.distributed
torch.distributed.is_available = lambda: False
torch.distributed.is_initialized = lambda: False

# Now import the model and utilities from train_gpt
sys.path.insert(0, '.')

# Load data
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

# Import model architecture from train_gpt
from train_gpt import GPT, Hyperparameters, quantize_state_dict_int8

def main():
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True

    args = Hyperparameters()
    seq_len = args.train_seq_len  # 1024
    batch_tokens = 8192  # small batch for 12GB VRAM
    batch_seqs = batch_tokens // seq_len  # 8 sequences per batch

    print(f"=== Parameter Golf Local Test ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Model: {args.num_layers}L, dim={args.model_dim}, heads={args.num_heads}")
    print(f"Batch: {batch_tokens} tokens ({batch_seqs} seqs x {seq_len})")

    # Load tokenizer
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)

    # Load validation tokens
    val_files = sorted(glob.glob(args.val_files))
    print(f"Loading {len(val_files)} validation shards...")
    val_tokens = torch.cat([load_data_shard(Path(f)) for f in val_files]).contiguous()
    print(f"Validation tokens: {val_tokens.numel():,}")

    # Load training tokens
    train_files = sorted(glob.glob(args.train_files))
    print(f"Loading {len(train_files)} training shards...")
    train_tokens = torch.cat([load_data_shard(Path(f)) for f in train_files]).contiguous()
    print(f"Training tokens: {train_tokens.numel():,}")

    # Build model
    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Simple AdamW optimizer (no Muon since it needs distributed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    # BPB lookup tables
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)

    # --- EVALUATE UNTRAINED ---
    def evaluate():
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        total_bytes = 0
        n_seqs = min((val_tokens.numel() - 1) // seq_len, 200)  # cap for speed
        with torch.inference_mode():
            for i in range(0, n_seqs, batch_seqs):
                end = min(i + batch_seqs, n_seqs)
                actual_batch = end - i
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

    print("\n--- Evaluating untrained model ---")
    loss0, bpb0 = evaluate()
    print(f"Untrained: val_loss={loss0:.4f}, val_bpb={bpb0:.4f}")

    # --- TRAIN ---
    num_iters = 100
    pos = 0
    print(f"\n--- Training for {num_iters} iterations ---")
    t0 = time.time()
    for step in range(num_iters):
        # Get batch
        need = batch_seqs * seq_len + 1
        if pos + need > train_tokens.numel():
            pos = 0
        chunk = train_tokens[pos:pos + need].to(device=device, dtype=torch.int64)
        pos += need
        x = chunk[:-1].reshape(-1, seq_len)
        y = chunk[1:].reshape(-1, seq_len)

        # Forward + backward
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 20 == 0:
            elapsed = time.time() - t0
            tps = (step + 1) * batch_tokens / elapsed
            print(f"  step {step+1}/{num_iters}: loss={loss.item():.4f}, {tps:.0f} tok/s, {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed:.1f}s")

    # --- EVALUATE TRAINED ---
    print("\n--- Evaluating trained model ---")
    loss1, bpb1 = evaluate()
    print(f"Trained:   val_loss={loss1:.4f}, val_bpb={bpb1:.4f}")
    print(f"Improvement: val_loss {loss0-loss1:+.4f}, val_bpb {bpb0-bpb1:+.4f}")

    # --- QUANTIZE AND MEASURE SIZE ---
    print("\n--- Quantizing model ---")
    obj, stats = quantize_state_dict_int8(model.state_dict())
    buf = io.BytesIO()
    torch.save(obj, buf)
    raw_bytes = buf.tell()
    compressed = zlib.compress(buf.getvalue(), 9)
    code_size = len(Path("train_gpt.py").read_bytes())
    total_artifact = len(compressed) + code_size
    print(f"Model params:    {stats['param_count']:,}")
    print(f"Raw int8 bytes:  {raw_bytes:,}")
    print(f"Compressed:      {len(compressed):,}")
    print(f"Code size:       {code_size:,}")
    print(f"Total artifact:  {total_artifact:,} bytes ({total_artifact/1e6:.2f} MB)")
    print(f"Under 16MB:      {'YES' if total_artifact <= 16_000_000 else 'NO'}")

    print(f"\n=== SUMMARY ===")
    print(f"Baseline BPB to beat: 1.2244 (naive baseline)")
    print(f"Current SOTA:          1.1233")
    print(f"Your local BPB:        {bpb1:.4f}")
    print(f"GPU:                   RTX 4070 SUPER")
    print(f"Artifact size:         {total_artifact/1e6:.2f} MB")

if __name__ == "__main__":
    main()
