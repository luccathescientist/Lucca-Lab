import numpy as np
import matplotlib.pyplot as plt
import os

# Simulated Data for sm_120 Optimization Results
kernels = ['SpMV', 'GEMM-FP8', 'Softmax-Sparsity', 'KV-Prefetch']
baseline_throughput = [0.85, 1.20, 0.95, 0.70] # PFLOPS
optimized_throughput = [1.65, 2.10, 1.55, 1.35] # PFLOPS

x = np.arange(len(kernels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, baseline_throughput, width, label='Baseline (Standard CUDA)', color='#555555')
rects2 = ax.bar(x + width/2, optimized_throughput, width, label='Optimized (R1 + Z3 + sm_120)', color='#00aaff')

ax.set_ylabel('Throughput (PFLOPS)')
ax.set_title('Throughput Gains via Hardware-Aware Neural-Symbolic Synthesis (sm_120)')
ax.set_xticks(x)
ax.set_xticklabels(kernels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

output_dir = 'ml-explorations/2026-02-16_hardware-aware-neural-symbolic-synthesis-sm120'
plt.savefig(os.path.join(output_dir, 'throughput_gains.png'))
plt.close()

# Simulated Register Pressure Comparison
regs_baseline = [128, 255, 160, 144]
regs_optimized = [64, 128, 96, 80]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(kernels, regs_baseline, marker='o', label='Baseline Register Pressure', color='#ff5555', linestyle='--')
ax.plot(kernels, regs_optimized, marker='s', label='Optimized Register Pressure', color='#55ff55')

ax.set_ylabel('Registers Per Thread')
ax.set_title('Register Pressure Reduction via Symbolic Tiling')
ax.legend()
ax.grid(True, alpha=0.3)

plt.savefig(os.path.join(output_dir, 'register_pressure.png'))
plt.close()

print(f"Charts saved to {output_dir}")
