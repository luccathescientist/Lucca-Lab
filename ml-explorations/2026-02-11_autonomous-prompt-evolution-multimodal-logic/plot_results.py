import matplotlib.pyplot as plt
import json

with open("evolution_results.json", "r") as f:
    data = json.load(f)

scores = [x[1] for x in data]
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(scores) + 1), scores, marker='o', linestyle='-', color='b')
plt.title("Autonomous Prompt Evolution: Spatial Reasoning Score")
plt.xlabel("Generation")
plt.ylabel("Performance Score (Simulated)")
plt.grid(True)
plt.savefig("performance_chart.png")
print("Chart generated.")
