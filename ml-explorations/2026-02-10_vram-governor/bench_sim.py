import matplotlib.pyplot as plt
import numpy as np

# Simulation data: Unmanaged vs Managed VRAM over a 3-stage pipeline
stages = ['Idle', 'DeepSeek-R1', 'Flux Generation', 'Wan Video Gen']
unmanaged = [2, 36, 48, 76]  # Accumulative OOM risk
managed = [2, 36, 42, 70]    # Proactive flushing

x = np.arange(len(stages))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, unmanaged, width, label='Unmanaged (Baseline)', color='#ff4c4c')
rects2 = ax.bar(x + width/2, managed, width, label='Managed (Lucca Governor)', color='#00cccc')

ax.set_ylabel('VRAM Usage (GB)')
ax.set_title('Blackwell RTX 6000: VRAM Governance Simulation')
ax.set_xticks(x)
ax.set_xticklabels(stages)
ax.axhline(y=96, color='r', linestyle='--', label='Hardware Limit (96GB)')
ax.legend()

plt.savefig('vram_benchmark.png')
print("Chart generated: vram_benchmark.png")
