import numpy as np
import matplotlib.pyplot as plt
import os

# Simulated Data for Autonomous CUDA Kernel Repair via Symbolic Execution
# Metrics: Register Pressure, Throughput (TFLOPS), and Memory Safety Score (0-100)

versions = ['Baseline', 'R1-Repaired (V1)', 'Symbolic-Verified (V2)']
throughput = [842, 1245, 1658] # TFLOPS
register_pressure = [255, 128, 96] # Registers per thread (Lower is better)
safety_score = [62, 88, 100] # OOB/Race-condition prevention

output_dir = 'ml-explorations/2026-02-16_autonomous-cuda-kernel-repair-symbolic-execution'

def plot_throughput():
    plt.figure(figsize=(10, 6))
    plt.bar(versions, throughput, color=['#ff9999','#66b3ff','#99ff99'])
    plt.title('Throughput Gain: Baseline vs. Symbolic-Verified Kernels (sm_120)')
    plt.ylabel('TFLOPS')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'throughput.png'))
    plt.close()

def plot_pressure():
    plt.figure(figsize=(10, 6))
    plt.plot(versions, register_pressure, marker='o', linestyle='-', color='orange', linewidth=2)
    plt.title('Register Pressure Reduction via Symbolic Tiling Optimization')
    plt.ylabel('Registers per Thread')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'register_pressure.png'))
    plt.close()

def plot_safety():
    plt.figure(figsize=(10, 6))
    plt.bar(versions, safety_score, color=['#e74c3c','#f1c40f','#2ecc71'])
    plt.title('Memory Safety Score (Symbolic Execution vs. Heuristic)')
    plt.ylabel('Safety Score (Percentage)')
    plt.ylim(0, 110)
    plt.savefig(os.path.join(output_dir, 'safety_score.png'))
    plt.close()

if __name__ == "__main__":
    plot_throughput()
    plot_pressure()
    plot_safety()
    print("Charts generated successfully.")
