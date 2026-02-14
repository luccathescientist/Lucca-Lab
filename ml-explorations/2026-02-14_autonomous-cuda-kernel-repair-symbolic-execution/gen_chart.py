import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ['Base R1', 'Neural-Symbolic (Previous)', 'Symbolic Verification (Current)']
oob_errors = [12, 4, 0]
throughput_utilization = [85, 94, 92] # Percentage of peak

x = np.arange(len(methods))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Method')
ax1.set_ylabel('OOB Errors (per 100 kernels)', color=color)
ax1.bar(x - width/2, oob_errors, width, label='OOB Errors', color=color, alpha=0.7)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('L2 Cache Utilization (%)', color=color)
ax2.plot(x + width/2, throughput_utilization, color=color, marker='o', label='Throughput %')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Autonomous CUDA Kernel Repair: Safety vs Performance')
fig.tight_layout()
plt.savefig('ml-explorations/2026-02-14_autonomous-cuda-kernel-repair-symbolic-execution/performance_chart.png')
