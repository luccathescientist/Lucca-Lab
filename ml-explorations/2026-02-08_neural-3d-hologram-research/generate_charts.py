import matplotlib.pyplot as plt
import numpy as np

# Mock data for Blackwell GPU metrics during the 3D Hologram test
time = np.linspace(0, 10, 100)
gpu_load = 15 + 5 * np.sin(time)  # Lightweight rendering
vram_usage = [24.5] * 100  # Baseline VRAM (Models resident)

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:cyan'
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('GPU Load (%)', color=color)
ax1.plot(time, gpu_load, color=color, label='GPU Load')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0, 100)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('VRAM Usage (GB)', color=color)
ax2.plot(time, vram_usage, color=color, linestyle='--', label='VRAM Usage')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 96)

plt.title('Blackwell Performance: Neural 3D Hologram (Three.js)')
fig.tight_layout()
plt.savefig('ml-explorations/2026-02-08_neural-3d-hologram-research/blackwell_metrics.png')
print("Chart generated: blackwell_metrics.png")
