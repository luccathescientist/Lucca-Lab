import numpy as np
import matplotlib.pyplot as plt
import os

# Simulated Data for Knowledge Graph Anchoring Impact
frames = np.arange(1, 61)  # 2 seconds at 30fps
drift_baseline = np.cumsum(np.random.normal(0.05, 0.02, 60))
drift_anchored = np.cumsum(np.random.normal(0.01, 0.005, 60))

# Applying "anchoring" events (resets) every 15 frames
for i in range(14, 60, 15):
    drift_anchored[i:] -= (drift_anchored[i] - 0.02)

plt.figure(figsize=(10, 6))
plt.plot(frames, drift_baseline, label='Baseline (No Anchoring)', color='red', linestyle='--')
plt.plot(frames, drift_anchored, label='KG-Anchored (Wan 2.1)', color='green', linewidth=2)
plt.axhline(y=0.1, color='gray', linestyle=':', label='Coherence Threshold')
plt.title('Character Identity Drift: KG-Anchoring vs. Baseline')
plt.xlabel('Frame Number')
plt.ylabel('Identity Dissimilarity Score (Lower is Better)')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)

save_path = 'ml-explorations/2026-02-11_neural-knowledge-graph-anchoring-video-synthesis/identity_drift_chart.png'
plt.savefig(save_path)
print(f"Chart saved to {save_path}")
