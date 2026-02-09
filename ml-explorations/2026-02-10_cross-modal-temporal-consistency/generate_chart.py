import numpy as np
import matplotlib.pyplot as plt

# Simulate Wan 2.1 frame-to-frame feature drift with and without 3D-UNet correction
frames = np.arange(0, 31)
# Feature stability (higher is better)
baseline_drift = 0.95 * np.exp(-0.02 * frames) + np.random.normal(0, 0.01, len(frames))
corrected_drift = 0.95 * np.exp(-0.005 * frames) + np.random.normal(0, 0.005, len(frames))

plt.figure(figsize=(10, 6))
plt.plot(frames, baseline_drift, label='Baseline Wan 2.1', color='red', linestyle='--')
plt.plot(frames, corrected_drift, label='Wan 2.1 + 3D-UNet Correction', color='cyan')
plt.title('Character Feature Consistency Over 30 Frames (720p)')
plt.xlabel('Frame Index')
plt.ylabel('Feature Similarity Score')
plt.ylim(0.8, 1.0)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()
plt.savefig('ml-explorations/2026-02-10_cross-modal-temporal-consistency/consistency_chart.png')
print("Chart generated.")
