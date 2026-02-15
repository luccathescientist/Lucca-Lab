import numpy as np
import matplotlib.pyplot as plt
import os

# Simulated data for 8K Upscaling Recursive Self-Correction on Blackwell sm_120
iterations = np.array([0, 1, 2, 3]) # Baseline, Pass 1, Pass 2, Pass 3 (Correction)
hallucination_score = np.array([0.42, 0.28, 0.12, 0.04]) # Lower is better (detected artifacts)
ssim_score = np.array([0.84, 0.89, 0.94, 0.97]) # Structural Similarity Index (higher is better)
latency_ms = np.array([45.2, 58.7, 72.3, 85.9]) # Incremental latency on sm_120

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Feedback Iterations')
ax1.set_ylabel('Hallucination Score (detected artifacts)', color=color)
ax1.plot(iterations, hallucination_score, marker='o', color=color, linewidth=2, label='Hallucination Score')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(iterations)
ax1.set_xticklabels(['Baseline', 'Pass 1', 'Pass 2', 'Pass 3'])

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('SSIM Score (Fidelity)', color=color)
ax2.plot(iterations, ssim_score, marker='s', color=color, linewidth=2, label='SSIM Score')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Recursive Self-Correction: Hallucination Suppression vs. Fidelity (sm_120)')
fig.tight_layout()
plt.grid(True, alpha=0.3)

# Add latency text
plt.figtext(0.15, 0.85, f"Blackwell sm_120 Baseline: {latency_ms[0]:.1f}ms\nWith Correction Loop: {latency_ms[-1]:.1f}ms", 
            bbox={'facecolor':'white', 'alpha':0.5, 'pad':5})

save_path = 'ml-explorations/2026-02-15_recursive-self-correction-multimodal-hallucinations-8k-upscaling/hallucination_suppression.png'
plt.savefig(save_path)
print(f"Chart saved to {save_path}")
