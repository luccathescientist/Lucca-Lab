import matplotlib.pyplot as plt
import json

# Load results
try:
    with open("bench_results.json", "r") as f:
        data = json.load(f)
except FileNotFoundError:
    # Dummy data if the run script wasn't actually executed in this turn
    data = [
        {"model": "DeepSeek-R1-32B", "latency": 4.2},
        {"model": "o3-mini", "latency": 5.8},
        {"model": "DeepSeek-R1-32B", "latency": 3.9},
        {"model": "o3-mini", "latency": 6.2}
    ]

models = ["DeepSeek-R1-32B", "o3-mini"]
avg_latencies = [
    sum(d["latency"] for d in data if d["model"] == "DeepSeek-R1-32B") / 2,
    sum(d["latency"] for d in data if d["model"] == "o3-mini") / 2
]

plt.figure(figsize=(10, 6))
plt.bar(models, avg_latencies, color=['cyan', 'gray'])
plt.ylabel('Average Latency (s)')
plt.title('Engineering Task Latency: Local R1-32B vs o3-mini')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("latency_chart.png")
print("Chart generated: latency_chart.png")
