#!/bin/bash
# Parameter Golf — Automated GPU Test Script
# This runs on the RunPod pod automatically after boot
# Results are saved to /runpod-volume/ for retrieval

set -e
echo "=========================================="
echo "  PARAMETER GOLF — GPU TEST RUNNER"
echo "=========================================="
echo "Timestamp: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "CUDA: $(nvcc --version 2>/dev/null | tail -1 || echo 'unknown')"

# Install dependencies
echo ""
echo "=== Installing dependencies ==="
pip install -q torch sentencepiece numpy 2>&1 | tail -5

# Clone repo
echo ""
echo "=== Cloning repo ==="
cd /workspace
if [ -d "parameter-golf-caum" ]; then
    cd parameter-golf-caum && git pull
else
    git clone https://github.com/caum-systems/parameter-golf-caum.git
    cd parameter-golf-caum
fi

# Run all verification tests
echo ""
echo "=========================================="
echo "  TEST 1: Optimizations Patch Verification"
echo "=========================================="
python optimizations_patch.py 2>&1

echo ""
echo "=========================================="
echo "  TEST 2: CAUM Warmdown Scheduler"
echo "=========================================="
python caum_warmdown.py 2>&1

echo ""
echo "=========================================="
echo "  TEST 3: Depth Recurrence Param Estimates"
echo "=========================================="
python depth_recurrence_patch.py 2>&1

echo ""
echo "=========================================="
echo "  TEST 4: GPU Tensor Operations"
echo "=========================================="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
    
    # Test BF16 matmul performance
    import time
    dim = 768
    x = torch.randn(32, 2048, dim, device='cuda', dtype=torch.bfloat16)
    w = torch.randn(dim, dim*3, device='cuda', dtype=torch.bfloat16)
    
    # Warmup
    for _ in range(10):
        y = x @ w
    torch.cuda.synchronize()
    
    t0 = time.time()
    for _ in range(100):
        y = x @ w
    torch.cuda.synchronize()
    t1 = time.time()
    
    tflops = (100 * 2 * 32 * 2048 * dim * dim * 3) / (t1 - t0) / 1e12
    print(f'BF16 matmul TFLOPS: {tflops:.1f}')
    print(f'  (H100 SXM expected: ~900-1000 TFLOPS)')
    print(f'  (A100 PCIe expected: ~300 TFLOPS)')
    
    # Test LoRA adapter speed
    from depth_recurrence_patch import LoRAAdapter
    adapter = LoRAAdapter(dim, dim, rank=8).cuda().bfloat16()
    bank_w = torch.randn(dim, dim, device='cuda', dtype=torch.bfloat16)
    
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(1000):
        adapted = adapter(bank_w)
    torch.cuda.synchronize()
    t1 = time.time()
    
    print(f'LoRA adapter: {1000/(t1-t0):.0f} calls/sec ({(t1-t0)/1000*1e6:.1f} µs/call)')
" 2>&1

echo ""
echo "=========================================="
echo "  TEST 5: Quick Training Loop (50 steps)"
echo "=========================================="
python -c "
import torch, torch.nn as nn, torch.nn.functional as F, time, math
from depth_recurrence_patch import LoRAAdapter
from optimizations_patch import scale_shared_bank_gradients, get_palindromic_bank_indices

# Mini model test (no dataset needed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dim, vocab, K, R, seq_len, batch = 256, 1024, 4, 3, 512, 8

# Create mini bank-based model
class MiniRecurrentGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, dim)
        self.qo_bank = nn.Parameter(torch.randn(2*K, dim, dim) * 0.02)
        self.mlp_up_bank = nn.Parameter(torch.randn(K, dim*3, dim) * 0.02)
        self.mlp_down_bank = nn.Parameter(torch.randn(K, dim, dim*3) * 0.02)
        self.lora_q = nn.ModuleList([LoRAAdapter(dim, dim, 8) for _ in range(K*R)])
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.tok_emb.weight  # tie embeddings
    
    def forward(self, x_ids, targets):
        x = self.tok_emb(x_ids)
        indices = get_palindromic_bank_indices(K, R)
        
        for eff_i, bi in enumerate(indices):
            # Attention shortcut (just linear projection)
            q_w = self.lora_q[eff_i](self.qo_bank[bi])
            x = x + F.linear(F.rms_norm(x, (dim,)), q_w.to(x.dtype)) * (1.0/math.sqrt(eff_i+1))
            
            # MLP
            up = F.linear(F.rms_norm(x, (dim,)), self.mlp_up_bank[bi].to(x.dtype))
            up = F.leaky_relu(up, 0.5) ** 2
            down = F.linear(up, self.mlp_down_bank[bi].to(x.dtype))
            x = x + down * (1.0/math.sqrt(eff_i+1))
        
        x = F.rms_norm(x, (dim,))
        logits = F.linear(x, self.tok_emb.weight)
        return F.cross_entropy(logits.view(-1, vocab), targets.view(-1))

model = MiniRecurrentGPT().to(device).bfloat16()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
n_params = sum(p.numel() for p in model.parameters())
print(f'Mini model: {n_params:,} params on {device}')
print(f'Config: K={K} R={R} dim={dim} palindromic routing')

losses = []
t0 = time.time()
for step in range(50):
    x = torch.randint(0, vocab, (batch, seq_len), device=device)
    y = torch.randint(0, vocab, (batch, seq_len), device=device)
    
    loss = model(x, y)
    loss.backward()
    
    # Apply Muon gradient scaling
    scale_shared_bank_gradients(model, R)
    
    opt.step()
    opt.zero_grad()
    
    losses.append(loss.item())
    if step % 10 == 0:
        print(f'  Step {step:3d}: loss={loss.item():.4f}')

elapsed = time.time() - t0
print(f'\\n  50 steps in {elapsed:.2f}s ({elapsed/50*1000:.1f} ms/step)')
print(f'  Loss: {losses[0]:.4f} -> {losses[-1]:.4f} (delta: {losses[-1]-losses[0]:.4f})')
print(f'  Training working correctly: {\"YES\" if losses[-1] < losses[0] else \"CHECK!\"}')
" 2>&1

echo ""
echo "=========================================="
echo "  ALL TESTS COMPLETE"
echo "=========================================="
echo "Timestamp: $(date)"
echo "DONE" > /workspace/test_complete.txt
