import numpy as np
import matplotlib.pyplot as plt
import os

# Simulated Data for Neural Knowledge Graph Anchoring (v2)
# Comparing accuracy and attention focus between baseline and anchored models

domains = ['CUDA Synthesis', 'Hardware Spec', 'Bit-Slicing', 'MoE Routing', 'Thermal Modeling']
baseline_accuracy = [74, 68, 62, 79, 71]
anchored_v2_accuracy = [92, 94, 88, 95, 91]

x = np.arange(len(domains))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, baseline_accuracy, width, label='Baseline R1', color='#4a4a4a')
rects2 = ax.bar(x + width/2, anchored_v2_accuracy, width, label='Anchored R1 (v2)', color='#00aaff') # Custom Lucca color

ax.set_ylabel('Accuracy (%)')
ax.set_title('Reasoning Accuracy with Neural Knowledge Graph Anchoring (v2)')
ax.set_xticks(x)
ax.set_xticklabels(domains)
ax.legend()

# GPU Throughput Simulation on Blackwell sm_120
batches = [1, 8, 16, 32, 64, 128]
throughput_baseline = [12, 85, 160, 310, 580, 1100]
throughput_anchored = [11.5, 82, 155, 302, 565, 1080] # Slight overhead for KG lookup

plt.figure(figsize=(10, 6))
plt.plot(batches, throughput_baseline, marker='o', label='Baseline Throughput')
plt.plot(batches, throughput_anchored, marker='s', label='Anchored Throughput')
plt.xlabel('Batch Size')
plt.ylabel('Tokens/Sec')
plt.title('Throughput vs. Batch Size (Blackwell sm_120)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Save charts
output_dir = 'ml-explorations/2026-02-15_neural-knowledge-graph-anchoring-v2'
plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'))
plt.savefig(os.path.join(output_dir, 'throughput_analysis.png'))

print("Charts generated successfully.")
