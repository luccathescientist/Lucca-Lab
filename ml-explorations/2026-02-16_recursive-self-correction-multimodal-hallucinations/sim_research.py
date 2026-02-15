import matplotlib.pyplot as plt
import numpy as np

# Data simulation for 8K Wan 2.1 Recursive Self-Correction
labels = ['Baseline (Wan 2.1)', 'Recursive (Pass 1)', 'Recursive (Pass 2)']
artifact_counts = [1240, 312, 118]
ssim_scores = [0.84, 0.92, 0.97]
latency_overhead = [0, 8.5, 14.8] # ms

x = np.arange(len(labels))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Upscaling Stage')
ax1.set_ylabel('Hallucinated Artifact Count', color=color)
bars = ax1.bar(x - width/2, artifact_counts, width, label='Artifacts', color=color, alpha=0.7)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('SSIM Score', color=color)
ax2.plot(x + width/2, ssim_scores, color=color, marker='o', label='SSIM')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0.8, 1.0)

plt.title('Recursive Self-Correction Efficacy (8K Wan 2.1 on Blackwell)')
fig.tight_layout()
plt.savefig('ml-explorations/2026-02-16_recursive-self-correction-multimodal-hallucinations/performance_chart.png')
print("Chart generated.")
