import matplotlib.pyplot as plt
import csv

modes = []
speedups = []

with open("results.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        modes.append(row['Mode'])
        speedups.append(float(row['Speedup']))

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(modes, speedups, color=['gray', 'cyan', 'blue'])
plt.axhline(y=1.0, color='r', linestyle='--', label='Baseline')
plt.title('Blackwell Bit-Level Speculative Decoding Speedup', fontsize=14)
plt.ylabel('Speedup Factor', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("speedup_chart.png")
print("Chart saved: speedup_chart.png")
