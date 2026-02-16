import numpy as np
import matplotlib.pyplot as plt
import os

# Simulation parameters for Dynamic KV-Cache Tiering on Blackwell sm_120
context_lengths = [128, 256, 512, 1024] # in K tokens
saliency_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

# Modeled Latency (ms) based on L2 vs HBM3e bandwidth
# L2 bandwidth: ~5 TB/s, HBM3e: ~1.2 TB/s
def simulate_tiering(context_len, threshold):
    # Higher threshold = more offloading to HBM3e
    # Base latency assumes linear scaling with context length
    base_latency = context_len * 0.05 
    
    # Tiering gain: reducing L2 misses by keeping high-saliency tokens in L2
    # Efficiency is modeled as a function of saliency gating accuracy
    tiering_gain = (1.0 - threshold) * 0.4 # up to 40% reduction
    
    # Overhead of saliency calculation
    overhead = 0.5 
    
    return base_latency * (1.0 - tiering_gain) + overhead

results = {}
for cl in context_lengths:
    results[cl] = [simulate_tiering(cl, t) for t in saliency_thresholds]

# Generate Technical Chart
plt.figure(figsize=(10, 6))
for cl, latencies in results.items():
    plt.plot(saliency_thresholds, latencies, marker='o', label=f'{cl}K context')

plt.title('Dynamic KV-Cache Tiering Latency vs. Saliency Threshold (Blackwell sm_120)')
plt.xlabel('Saliency Gating Threshold (Offload Factor)')
plt.ylabel('Inference Latency (ms)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig('ml-explorations/2026-02-17_dynamic-kv-cache-tiering-hierarchical-blackwell-storage/tiering_performance.png')

# Output Raw Data for Reproducibility
print(f"Simulation results for Dynamic KV-Cache Tiering:")
for cl, latencies in results.items():
    print(f"Context {cl}K: {latencies}")
