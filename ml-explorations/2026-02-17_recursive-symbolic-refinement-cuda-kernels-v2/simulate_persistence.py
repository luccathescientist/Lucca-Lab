import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np

# Simulation of Blackwell (sm_120) Tensor Core Instruction: L2-resident weight persistence
# This script compares standard kernel execution vs. a simulated "L2-Persistence" optimized kernel.

class BlackwellSimulator:
    def __init__(self):
        self.l2_cache_size = 128 * 1024 * 1024 # 128MB
        self.hbm_bandwidth = 8.0 # TB/s (Simulated HBM3e)
        self.l2_bandwidth = 20.0 # TB/s (Simulated L2)

    def simulate_matmul(self, m, n, k, persistence=False):
        # Calculate weight size
        weight_size = k * n * 1 # Assuming 1 byte per element (FP8/INT8)
        
        # Data movement cost
        if persistence and weight_size <= self.l2_cache_size:
            # Weight is pinned in L2, only fetch activations and write outputs
            data_transferred = (m * k + m * n) * 1
            bandwidth = self.l2_bandwidth
        else:
            # Fetch weights, activations, and write outputs from HBM
            data_transferred = (m * k + k * n + m * n) * 1
            bandwidth = self.hbm_bandwidth
            
        latency = data_transferred / (bandwidth * 1e12)
        # Compute cost (simulated TFLOPS)
        compute_ops = 2 * m * n * k
        compute_latency = compute_ops / (2000 * 1e12) # 2 PFLOPS peak
        
        total_latency = max(latency, compute_latency) # Roofline model
        return total_latency

def run_experiment():
    sim = BlackwellSimulator()
    m, k = 4096, 4096
    n_sizes = [1024, 2048, 4096, 8192, 16384]
    
    standard_latencies = []
    persistent_latencies = []
    
    for n in n_sizes:
        standard_latencies.append(sim.simulate_matmul(m, n, k, persistence=False) * 1000) # ms
        persistent_latencies.append(sim.simulate_matmul(m, n, k, persistence=True) * 1000) # ms
        
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(n_sizes, standard_latencies, marker='o', label='Standard HBM-Fetch')
    plt.plot(n_sizes, persistent_latencies, marker='s', label='L2-Resident Weight Persistence')
    plt.xlabel('Inner Dimension (N)')
    plt.ylabel('Latency (ms)')
    plt.title('Blackwell sm_120: L2-Resident Weight Persistence Simulation')
    plt.legend()
    plt.grid(True)
    plt.savefig('ml-explorations/2026-02-17_recursive-symbolic-refinement-cuda-kernels-v2/performance_comparison.png')
    
    print(f"Experiment completed. Chart saved.")

if __name__ == "__main__":
    run_experiment()
