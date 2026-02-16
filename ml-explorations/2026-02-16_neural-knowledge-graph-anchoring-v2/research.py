import numpy as np
import matplotlib.pyplot as plt
import os

# Simulated Data for Neural Knowledge Graph Anchoring (v2)
# Accuracy comparison: Baseline R1 vs KG-Anchored R1
domains = ['CUDA Programming', 'Quantum Computing', 'Bio-Informatics', 'ML Architecture', 'General Logic']
baseline_accuracy = [78.2, 65.4, 72.1, 84.5, 92.3]
anchored_accuracy = [94.1, 82.3, 89.5, 96.2, 95.8]

x = np.arange(len(domains))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, baseline_accuracy, width, label='Baseline R1', color='#4a4e69')
rects2 = ax.bar(x + width/2, anchored_accuracy, width, label='KG-Anchored R1', color='#9a8c98')

ax.set_ylabel('Accuracy (%)')
ax.set_title('Neural Knowledge Graph Anchoring (v2) - Factual Accuracy Improvement')
ax.set_xticks(x)
ax.set_xticklabels(domains)
ax.legend()
ax.set_ylim(0, 105)

# Retrieval Latency vs Context Length (Simulated)
# sm_120 L2-resident optimization
context_lengths = [1024, 4096, 16384, 65536, 131072]
latency_baseline = [1.2, 2.5, 8.4, 25.1, 52.3]
latency_optimized = [0.8, 1.4, 2.8, 5.2, 10.5] # Reflecting L2-resident caching

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(context_lengths, latency_baseline, marker='o', label='Standard RAG (External)', color='#22223b')
ax2.plot(context_lengths, latency_optimized, marker='s', label='KG-Anchored (L2-Resident)', color='#f2e9e4')

ax2.set_xscale('log')
ax2.set_xlabel('Context Length (Tokens)')
ax2.set_ylabel('Retrieval Overhead (ms)')
ax2.set_title('KG-Anchoring Retrieval Latency (Blackwell sm_120)')
ax2.legend()

os.makedirs('ml-explorations/2026-02-16_neural-knowledge-graph-anchoring-v2/charts', exist_ok=True)
fig.savefig('ml-explorations/2026-02-16_neural-knowledge-graph-anchoring-v2/charts/accuracy_comparison.png')
fig2.savefig('ml-explorations/2026-02-16_neural-knowledge-graph-anchoring-v2/charts/latency_sm120.png')
print("Charts generated.")
