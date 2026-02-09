import matplotlib.pyplot as plt
import numpy as np

# Data
labels = ['Separate Kernels', 'Fused Kernel']
# N = 10^7 floats, 4 bytes each
# Bandwidth = 1.5 TB/s = 1500 GB/s
N = 10**7
float_size = 4 # bytes
bandwidth = 1500 * 10**9 # bytes/s

# 8N transfers for separate
# 3N transfers for fused
transfers_separate = 8 * N * float_size
transfers_fused = 3 * N * float_size

latency_separate = transfers_separate / bandwidth * 1000 # ms
latency_fused = transfers_fused / bandwidth * 1000 # ms

# Latencies
latencies = [latency_separate, latency_fused]

# Plot
plt.figure(figsize=(10, 6))
bars = plt.bar(labels, latencies, color=['#ff9999','#66b3ff'])
plt.ylabel('Latency (ms)')
plt.title('Theoretical Latency: Separate vs. Fused Kernels on Blackwell RTX 6000')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add values on top
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.001, f'{yval:.4f} ms', ha='center', va='bottom')

plt.savefig('Lucca-Lab/ml-explorations/2026-02-09_automated-kernel-fusion/latency_comparison.png')
print("Chart generated: latency_comparison.png")
