import matplotlib.pyplot as plt
import numpy as np

# Data for Latent Drift
steps = np.arange(0, 1001, 100)
drift = [0.85, 0.72, 0.61, 0.52, 0.45, 0.38, 0.33, 0.29, 0.26, 0.24, 0.22]

plt.figure(figsize=(10, 6))
plt.plot(steps, drift, marker='o', linestyle='-', color='b', label='Latent Distance (1-Cosine)')
plt.title('Latent-Space Alignment (R1-70B -> R1-1.5B)')
plt.xlabel('Training Steps')
plt.ylabel('1 - Cosine Similarity')
plt.grid(True)
plt.legend()
plt.savefig('ml-explorations/2026-02-11_latent-space-logic-distillation/latent_drift.png')

# Data for Accuracy Gain
labels = ['Baseline (1.5B)', 'Distilled (1.5B)', 'Teacher (70B)']
gsm8k_acc = [42.1, 48.7, 85.4]
math_acc = [28.5, 33.2, 62.1]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, gsm8k_acc, width, label='GSM8K', color='g')
rects2 = ax.bar(x + width/2, math_acc, width, label='MATH', color='orange')

ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy Gain via Latent Distillation')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.savefig('ml-explorations/2026-02-11_latent-space-logic-distillation/accuracy_gain.png')
