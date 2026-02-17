import matplotlib.pyplot as plt
import json

with open("results.json", "r") as f:
    data = json.load(f)

metrics = list(data.keys())
values = [data[m]["avg"] for m in metrics]

plt.figure(figsize=(10, 6))
plt.bar(metrics, values, color=['blue', 'green', 'red'])
plt.title("Multi-Agent Consensus Benchmark Results (Blackwell sm_120)")
plt.ylabel("Value")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("benchmark_chart.png")
print("Chart saved as benchmark_chart.png")
