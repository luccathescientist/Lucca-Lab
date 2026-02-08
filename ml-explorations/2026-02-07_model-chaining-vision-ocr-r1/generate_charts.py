import matplotlib.pyplot as plt

# Simulate latency components
labels = ['Vision (Llama-3)', 'OCR', 'Reasoning (R1)']
latencies = [1.2, 0.5, 2.8] # seconds

plt.figure(figsize=(10, 6))
plt.bar(labels, latencies, color=['#00cfd5', '#008b8b', '#00f5ff'])
plt.title('Pipeline Latency Breakdown (Blackwell RTX 6000)')
plt.ylabel('Latency (seconds)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('latency_chart.png')
print("Chart generated: latency_chart.png")
