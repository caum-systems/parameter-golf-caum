#!/bin/bash
# Parameter Golf — GPU Test Runner v2
# Saves results to /workspace/results.json for easy API retrieval
# AND to a simple HTTP file on the pod's Jupyter server

set -e
RESULTS_FILE="/workspace/results.json"
LOG_FILE="/workspace/gpu_test_log.txt"

# Redirect all output to log file AND stdout
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "  PARAMETER GOLF — GPU TEST v2"
echo "=========================================="
echo "Timestamp: $(date)"

# Get GPU info
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "unknown")
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo "unknown")
CUDA_VER=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' | sed 's/,//' || echo "unknown")
echo "GPU: $GPU_NAME ($GPU_MEM)"
echo "CUDA: $CUDA_VER"

# Install deps
echo ""
echo "=== Installing dependencies ==="
pip install -q sentencepiece 2>&1 | tail -2

# Clone/update repo
echo ""
echo "=== Setting up repo ==="
cd /workspace
if [ -d "parameter-golf-caum" ]; then
    cd parameter-golf-caum && git pull
else
    git clone https://github.com/caum-systems/parameter-golf-caum.git
    cd parameter-golf-caum
fi

# Run ALL tests and capture JSON results
echo ""
echo "=== Running tests ==="

python3 << 'PYTHON_SCRIPT'
import torch, torch.nn as nn, torch.nn.functional as F
import time, math, json, sys
sys.path.insert(0, '.')
from depth_recurrence_patch import LoRAAdapter
from optimizations_patch import scale_shared_bank_gradients, get_palindromic_bank_indices

results = {}

# === TEST 1: GPU Info ===
print("\n--- TEST 1: GPU Info ---")
results["gpu_name"] = torch.cuda.get_device_name(0)
results["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
results["pytorch"] = torch.__version__
results["cuda"] = torch.version.cuda
print(f"  GPU: {results['gpu_name']}")
print(f"  VRAM: {results['gpu_vram_gb']} GB")
print(f"  PyTorch: {results['pytorch']}")
print(f"  CUDA: {results['cuda']}")
results["test1"] = "PASS"

# === TEST 2: BF16 Benchmark ===
print("\n--- TEST 2: BF16 TFLOPS ---")
dim = 768
x = torch.randn(32, 2048, dim, device='cuda', dtype=torch.bfloat16)
w = torch.randn(dim, dim*3, device='cuda', dtype=torch.bfloat16)
for _ in range(10): y = x @ w
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100): y = x @ w
torch.cuda.synchronize()
elapsed = time.time() - t0
tflops = (100 * 2 * 32 * 2048 * dim * dim * 3) / elapsed / 1e12
results["bf16_tflops"] = round(tflops, 1)
results["matmul_time_ms"] = round(elapsed / 100 * 1000, 2)
print(f"  BF16 TFLOPS: {tflops:.1f}")
print(f"  Per matmul: {elapsed/100*1000:.2f}ms")
results["test2"] = "PASS"

# === TEST 3: LoRA Speed ===
print("\n--- TEST 3: LoRA Adapter Speed ---")
adapter = LoRAAdapter(dim, dim, rank=8).cuda().bfloat16()
bank_w = torch.randn(dim, dim, device='cuda', dtype=torch.bfloat16)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(1000): adapted = adapter(bank_w)
torch.cuda.synchronize()
lora_time = (time.time() - t0) / 1000 * 1e6
results["lora_us_per_call"] = round(lora_time, 1)
print(f"  LoRA: {lora_time:.1f} µs/call ({1e6/lora_time:.0f} calls/sec)")
results["test3"] = "PASS"

# === TEST 4: Mini Training Loop ===
print("\n--- TEST 4: Training Loop (50 steps) ---")
device = 'cuda'
dim, vocab, K, R, seq_len, batch = 256, 1024, 4, 3, 512, 8

class Mini(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, dim)
        self.mlp_up_bank = nn.Parameter(torch.randn(K, dim*3, dim) * 0.02)
        self.mlp_down_bank = nn.Parameter(torch.randn(K, dim, dim*3) * 0.02)
        self.qo_bank = nn.Parameter(torch.randn(2*K, dim, dim) * 0.02)
        self.kv_bank = nn.Parameter(torch.randn(2*K, dim, dim) * 0.02)
        self.lora_q = nn.ModuleList([LoRAAdapter(dim, dim, 8) for _ in range(K*R)])
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.tok_emb.weight
    def forward(self, x_ids, targets):
        x = self.tok_emb(x_ids)
        for eff_i, bi in enumerate(get_palindromic_bank_indices(K, R)):
            q_w = self.lora_q[eff_i](self.qo_bank[bi])
            x = x + F.linear(F.rms_norm(x, (dim,)), q_w.to(x.dtype)) * (1/math.sqrt(eff_i+1))
            up = F.linear(F.rms_norm(x, (dim,)), self.mlp_up_bank[bi].to(x.dtype))
            x = x + F.linear(F.leaky_relu(up, 0.5)**2, self.mlp_down_bank[bi].to(x.dtype)) * (1/math.sqrt(eff_i+1))
        return F.cross_entropy(F.linear(F.rms_norm(x,(dim,)), self.tok_emb.weight).view(-1,vocab), targets.view(-1))

model = Mini().to(device).bfloat16()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
n_params = sum(p.numel() for p in model.parameters())
results["mini_model_params"] = n_params

losses = []
t0 = time.time()
for s in range(50):
    x = torch.randint(0, vocab, (batch, seq_len), device=device)
    loss = model(x, x)
    loss.backward()
    scale_shared_bank_gradients(model, R)
    opt.step(); opt.zero_grad()
    losses.append(loss.item())
    if s % 10 == 0:
        print(f"  Step {s:3d}: loss={loss.item():.4f}")

elapsed = time.time() - t0
results["training_50steps_sec"] = round(elapsed, 2)
results["ms_per_step"] = round(elapsed / 50 * 1000, 1)
results["loss_start"] = round(losses[0], 4)
results["loss_end"] = round(losses[-1], 4)
results["loss_delta"] = round(losses[-1] - losses[0], 4)
results["training_ok"] = losses[-1] < losses[0]
print(f"\n  50 steps: {elapsed:.2f}s ({elapsed/50*1000:.1f} ms/step)")
print(f"  Loss: {losses[0]:.4f} → {losses[-1]:.4f} (delta: {losses[-1]-losses[0]:.4f})")
print(f"  Training OK: {results['training_ok']}")
results["test4"] = "PASS" if results["training_ok"] else "FAIL"

# === TEST 5: Optimizations verification ===
print("\n--- TEST 5: Optimizations Verified ---")
from optimizations_patch import get_palindromic_bank_indices
p = get_palindromic_bank_indices(6, 2)
results["palindromic_6x2"] = p
results["test5"] = "PASS" if p == [0,1,2,3,4,5,5,4,3,2,1,0] else "FAIL"
print(f"  Palindromic 6x2: {p}")

# === SUMMARY ===
results["all_pass"] = all(results.get(f"test{i}") == "PASS" for i in range(1,6))
results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

# Save JSON
with open("/workspace/results.json", "w") as f:
    json.dump(results, f, indent=2)

# Also save to a file accessible via Jupyter
with open("/workspace/results.json", "r") as f:
    print(f"\n{'='*50}")
    print("  RESULTS JSON (copy this if needed):")
    print(f"{'='*50}")
    print(f.read())

print(f"\n{'='*50}")
print(f"  ALL TESTS: {'✅ PASSED' if results['all_pass'] else '❌ SOME FAILED'}")
print(f"{'='*50}")
PYTHON_SCRIPT

echo ""
echo "=========================================="
echo "  TESTS COMPLETE — Results at /workspace/results.json"  
echo "  Fetch via: https://POD_ID-8888.proxy.runpod.net/files/results.json"
echo "=========================================="
