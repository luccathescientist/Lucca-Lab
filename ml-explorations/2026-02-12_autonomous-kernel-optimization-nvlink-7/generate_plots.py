import numpy as np
import matplotlib.pyplot as plt

# Simulated performance data for NVLink-7 vs NVLink-6
# Metrics: Bandwidth (GB/s), Latency (ns), Throughput (TFLOPS) for P2P copies

devices = ['NVLink-6', 'NVLink-7 (Baseline)', 'NVLink-7 (Optimized)']
bandwidth = [900, 1800, 1950]  # GB/s
latency = [120, 80, 65]        # ns

def generate_plots():
    # Bandwidth Comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(devices, bandwidth, color=['gray', 'blue', 'green'])
    plt.title('Interconnect Bandwidth Comparison (P2P)')
    plt.ylabel('Bandwidth (GB/s)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 20, yval, ha='center', va='bottom')
    plt.savefig('ml-explorations/2026-02-12_autonomous-kernel-optimization-nvlink-7/plots/bandwidth_comparison.png')
    plt.close()

    # Latency Comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(devices, latency, color=['gray', 'blue', 'green'])
    plt.title('Interconnect Latency Comparison (P2P)')
    plt.ylabel('Latency (ns)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 2, yval, ha='center', va='bottom')
    plt.savefig('ml-explorations/2026-02-12_autonomous-kernel-optimization-nvlink-7/plots/latency_comparison.png')
    plt.close()

if __name__ == "__main__":
    generate_plots()
    print("Plots generated successfully.")
