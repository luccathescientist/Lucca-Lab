import matplotlib.pyplot as plt

# Metrics for Neural Knowledge Fusion
labels = ['Vector Only', 'Knowledge Graph Only', 'Hybrid Fusion']
accuracy = [78, 84, 95]
latency = [45, 30, 65] # ms

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Retrieval Method')
ax1.set_ylabel('Accuracy (%)', color=color)
ax1.bar(labels, accuracy, color=color, alpha=0.6, label='Accuracy')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0, 100)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Latency (ms)', color=color)
ax2.plot(labels, latency, color=color, marker='o', label='Latency')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 100)

plt.title('Neural Knowledge Graph Fusion Performance')
fig.tight_layout()
plt.savefig('ml-explorations/2026-02-10_neural-knowledge-graph-fusion/results_chart.png')
