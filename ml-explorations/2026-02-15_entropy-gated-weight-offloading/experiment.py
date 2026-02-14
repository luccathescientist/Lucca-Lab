import time
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_simulated_entropy(noise_scale):
    # Simulated entropy matching the expected behavior
    return 4.0 + (noise_scale * 2.0) + np.random.normal(0, 0.2)

def run_simulated_experiment():
    threshold = 6.2
    # Latency values based on previous lab benchmarks for Blackwell NVMe-to-L2
    LATENCY_BASE = 1.2 # ms
    LATENCY_LOAD = 45.0 # ms (Simulated NVMe load)
    LATENCY_OFFLOAD = 0.5 # ms
    
    entropies = []
    latencies = []
    vram_usage = []
    
    heavy_loaded = False
    
    for i in range(50):
        noise_scale = 0.1 if i % 10 < 7 else 1.5
        entropy = calculate_simulated_entropy(noise_scale)
        
        current_latency = LATENCY_BASE
        
        if entropy > threshold:
            if not heavy_loaded:
                current_latency += LATENCY_LOAD
                heavy_loaded = True
            current_latency += 0.8 # Extra processing for heavy layer
        else:
            if heavy_loaded:
                current_latency += LATENCY_OFFLOAD
                heavy_loaded = False
        
        entropies.append(entropy)
        latencies.append(current_latency)
        vram_usage.append(128 if heavy_loaded else 12) # Simulated MB
        
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Inference Step')
    ax1.set_ylabel('Entropy', color=color)
    ax1.plot(entropies, color=color, label='Entropy')
    ax1.axhline(y=threshold, color='r', linestyle='--', label='Gating Threshold')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Latency (ms)', color=color)
    ax2.plot(latencies, color=color, label='Latency')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Entropy-Gated Weight Offloading Performance (Blackwell Simulation)')
    fig.tight_layout()
    plt.savefig('entropy_gating_perf.png')
    
    print(f"Mean Latency: {np.mean(latencies):.2f}ms")
    print(f"Max Latency (Load Event): {np.max(latencies):.2f}ms")
    print(f"Average VRAM Usage: {np.mean(vram_usage):.2f}MB")
    print(f"VRAM Saved: {128 - np.mean(vram_usage):.2f}MB (Avg)")

if __name__ == "__main__":
    run_simulated_experiment()
