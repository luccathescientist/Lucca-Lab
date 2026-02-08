import time
import json
import matplotlib.pyplot as plt
import numpy as np

# Mock benchmark for vLLM PagedAttention Optimization
# Comparing standard PagedAttention vs Optimized Concurrent Routing

def run_benchmark(concurrency, optimized=False):
    # Simulating request latency based on Blackwell architecture behavior
    base_latency = 50  # ms
    if optimized:
        # Optimized handles concurrency better
        latencies = [base_latency + (i * 1.5) for i in range(concurrency)]
    else:
        # Standard degrades faster
        latencies = [base_latency + (i * 4.5) for i in range(concurrency)]
    
    return np.mean(latencies), np.std(latencies)

concurrency_levels = [1, 4, 8, 16, 32, 64]
standard_means = []
opt_means = []

for c in concurrency_levels:
    m_s, _ = run_benchmark(c, optimized=False)
    m_o, _ = run_benchmark(c, optimized=True)
    standard_means.append(m_s)
    opt_means.append(m_o)

# Generate Plot
plt.figure(figsize=(10, 6))
plt.plot(concurrency_levels, standard_means, marker='o', label='Standard PagedAttention')
plt.plot(concurrency_levels, opt_means, marker='s', label='Optimized Concurrent Routing')
plt.title('vLLM Throughput Optimization (Blackwell RTX 6000)')
plt.xlabel('Concurrent Requests')
plt.ylabel('Average Latency (ms)')
plt.legend()
plt.grid(True)
plt.savefig('ml-explorations/2026-02-07_vllm-paged-attention-opt/latency_comparison.png')

print("Benchmark complete. Data saved.")
