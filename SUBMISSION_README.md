# CAUM Systems — Parameter Golf Submission

## Architecture: Depth Recurrence with SOTA Optimizations

**val_bpb: TBD** (pending 8×H100 run) | **~6-8 MB** estimated | 8×H100 SXM

## Key Innovation: Depth Recurrence

Instead of N unique transformer layers (N sets of weights), we use K unique blocks
applied R times each. This gives R×K effective depth with only K blocks of parameters.

This approach was validated by two independent experiments:
- **Agent A**: rec_6×2@640 (6 blocks × 2 passes) — BPB=2.2691 at 500 steps
- **Agent B**: rec_4×3@512 (4 blocks × 3 passes) — loss=4.0944 at 500 steps

Both confirmed depth recurrence learns faster than standard architectures at equal step count.

### Why This Works

Standard 11-layer GPT: 11 unique weight matrices → 11× the storage
Depth Recurrence 6×2: 6 unique weight matrices × 2 passes = 12 effective depth → 6× storage

The saved parameter budget allows:
- Wider model dimension (640 vs 512)
- Better quantization (int6 instead of int8)
- More headroom for test-time training weights

## CAUM Integration

This submission was developed with assistance from CAUM Systems'
complexity analysis engine. Specifically:

### LZ76 Complexity Scoring for Data Analysis
CAUM's LZ76 compression-based complexity metric was used to analyze
12,207 training chunks from FineWeb. Key findings:
- Compression ratios ranged from 0.397 to 0.836 (mean: 0.721)
- Easy-first curriculum (CAUM-ordered) matched random ordering
- **Conclusion**: FineWeb is already well-curated for language modeling

### Architecture Search Guided by Trajectory Analysis
The experiment pipeline used CAUM's trajectory analysis concepts to
evaluate convergence speed across 15+ architecture configurations,
identifying depth recurrence as the optimal approach.

## Architecture Details

| Component | Setting |
|-----------|---------|
| Unique Blocks | 6 (applied 2× each = 12 effective depth) |
| Dimension | 640 |
| Heads | 8 (head_dim=80) |
| KV Heads | 4 (GQA) |
| MLP | 3× with LeakyReLU(0.5)² |
| BigramHash | 1536 buckets, 128-dim |
| Embeddings | Tied (tok_emb = lm_head) |
| Logit Softcap | 30.0 |
| RoPE Base | 10,000 |

## Training Configuration

| Setting | Value |
|---------|-------|
| Optimizer | Muon (Newton-Schulz) + AdamW (embeddings) |
| Weight Averaging | EMA(0.997) + SWA |
| Quantization | int8 + zlib (targeting int6 + lzma) |
| Max Steps | ~14,000 (10 min wallclock) |

## Experiments Conducted

| Round | Best Config | BPB/Loss | Key Finding |
|-------|-------------|----------|-------------|
| 1. Architecture | 11L/3x MLP | 2.3551 | Deeper + wider MLP is better |
| 2. Enhancements | + LeakyReLU² + BigramHash | 2.3427 | BigramHash gives −0.0455 |
| 3. Moonshots | **Depth Recurrence 6×2@640** | **2.2691** | Weight sharing = breakthrough |
| 4. CAUM Curriculum | LZ76 easy-first | 2.2601 | FineWeb already curated |

## Team

- **Andres Silva** — CAUM Systems (caum-systems)
- **Architecture Search**: Automated via independent experiment agents
- **Monitoring**: CAUM v10.31.0 trajectory analysis

## Links

- Competition: https://openai.com/index/parameter-golf/
- CAUM Systems: https://github.com/caum-systems
- This repo: https://github.com/caum-systems/parameter-golf-caum
