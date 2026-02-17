import numpy as np
import matplotlib.pyplot as plt
import os

# Simulated data for Hardware-Aware Neural-Symbolic Synthesis (v2)
# Comparing standard Triton kernels vs. Neural-Symbolic Optimized (INT3/sm_120)

configs = ['Standard Triton', 'R1-Symbolic (v1)', 'R1-Symbolic (v2) - sm_120']
throughput = [1.2, 1.65, 2.15]  # PFLOPS
latency = [24.5, 18.2, 14.1]    # ms (per 1k tokens)
register_pressure = [96, 48, 32] # register usage

# Create plot
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Kernel Synthesis Strategy')
ax1.set_ylabel('Throughput (PFLOPS)', color=color)
ax1.bar(configs, throughput, color=color, alpha=0.6, label='Throughput')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Latency (ms)', color=color)
ax2.plot(configs, latency, color=color, marker='o', linewidth=2, label='Latency')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Performance Gains: Neural-Symbolic Synthesis for sm_120 (v2)')
fig.tight_layout()

# Save plot
output_path = 'ml-explorations/2026-02-17_hardware-aware-neural-symbolic-synthesis-sm120-v2/performance_chart.png'
plt.savefig(output_path)
print(f"Chart saved to {output_path}")
