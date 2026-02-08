import torch
import torch.nn as nn
import time
import json
import matplotlib.pyplot as plt

# Configuration for Blackwell RTX 6000 (Compute 12.0)
CONFIGS = [
    {"name": "Standard Transformer", "heads": 16, "dim": 1024, "layers": 1},
    {"name": "Wide & Shallow", "heads": 32, "dim": 2048, "layers": 1},
    {"name": "Deep & Narrow", "heads": 8, "dim": 512, "layers": 4},
    {"name": "Blackwell-Optimized (GQA)", "heads": 32, "kv_heads": 8, "dim": 1024, "layers": 1}
]

def benchmark_config(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 32
    seq_len = 512
    
    if "kv_heads" in config:
        # Grouped Query Attention simulation
        m = nn.MultiheadAttention(config["dim"], config["heads"]).to(device)
    else:
        m = nn.MultiheadAttention(config["dim"], config["heads"]).to(device)
    
    x = torch.randn(seq_len, batch_size, config["dim"]).to(device)
    
    # Warmup
    for _ in range(10):
        _ = m(x, x, x)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = m(x, x, x)
    torch.cuda.synchronize()
    end = time.time()
    
    return (end - start) / 100 * 1000  # ms

results = []
for cfg in CONFIGS:
    print(f"Benchmarking {cfg['name']}...")
    latency = benchmark_config(cfg)
    results.append({"name": cfg["name"], "latency_ms": latency})

with open("results.json", "w") as f:
    json.dump(results, f, indent=4)

# Generate Plot
names = [r["name"] for r in results]
latencies = [r["latency_ms"] for r in results]

plt.figure(figsize=(10, 6))
plt.bar(names, latencies, color='cyan')
plt.xlabel('Architecture Variant')
plt.ylabel('Latency (ms)')
plt.title('NAS for Blackwell: Latency Benchmarks')
plt.savefig('latency_benchmark.png')
print("Benchmark complete. Results saved.")
