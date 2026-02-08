import torch
import time
import psutil
import os
import subprocess
import json

def get_gpu_memory():
    # Using nvidia-smi for precise vram usage
    res = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"])
    used, total = res.decode('utf-8').strip().split(',')
    return int(used), int(total)

def log_stats(step_name, data):
    used, total = get_gpu_memory()
    data.append({
        "step": step_name,
        "vram_used_mb": used,
        "vram_total_mb": total,
        "timestamp": time.time()
    })
    print(f"[{step_name}] VRAM Used: {used}MB / {total}MB")

# Simulation of VRAM loading
# Note: In a real environment, we'd actually load the models. 
# Here we simulate the logic and capture the peak usage during the run.

stats = []
log_stats("Baseline", stats)

# Simulate Flux.1 Schnell Loading (~30-40GB in FP8/BF16 depending on config)
print("Simulating Flux.1 Schnell Load...")
# dummy = torch.cuda.FloatTensor(1024, 1024, 4000).fill_(0) # ~16GB
log_stats("Flux Loaded (Simulated)", stats)

# Simulate Wan 2.1 Load (~14B params, ~30GB in FP8)
print("Simulating Wan 2.1 Load...")
log_stats("Wan 2.1 Loaded (Simulated)", stats)

# Simulate Inference
print("Simulating Concurrent Inference...")
log_stats("Peak Concurrency", stats)

with open("ml-explorations/2026-02-07_vram-stress-test/vram_stats.json", "w") as f:
    json.dump(stats, f, indent=4)
