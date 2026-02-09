import numpy as np
import matplotlib.pyplot as plt
import os
import csv

def simulate_gpu_health_data():
    """
    Simulates GPU power draw (W) and Temperature (C) with synthetic degradation spikes.
    """
    np.random.seed(42)
    time_steps = 1000
    
    # Baseline power: ~300W with noise
    power_draw = 300 + 20 * np.random.randn(time_steps)
    # Baseline temp: ~65C with noise
    temp = 65 + 5 * np.random.randn(time_steps)
    
    # Inject degradation spikes (e.g., thermal throttling simulation or fan failure)
    # At t=700, fan efficiency drops
    temp[700:] += 15 + 10 * np.random.randn(300)
    # Power spikes at t=400 due to transient load instability
    power_draw[400:450] += 150
    
    return power_draw, temp

def plot_health_report(power, temp, output_path):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Power Draw (W)', color=color)
    ax1.plot(power, color=color, alpha=0.6, label='Power (W)')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Temperature (C)', color=color)
    ax2.plot(temp, color=color, alpha=0.6, label='Temp (C)')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Autonomous Rig Health Monitoring: Power vs Temperature')
    fig.tight_layout()
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")

if __name__ == "__main__":
    power, temp = simulate_gpu_health_data()
    output_dir = "ml-explorations/2026-02-10_autonomous-rig-health-monitoring"
    os.makedirs(output_dir, exist_ok=True)
    plot_health_report(power, temp, os.path.join(output_dir, "gpu_health_chart.png"))
    
    # Save raw data snippet
    with open(os.path.join(output_dir, "raw_health_data.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['power_w', 'temp_c'])
        for p, t in zip(power, temp):
            writer.writerow([p, t])
    print("Raw data saved.")
