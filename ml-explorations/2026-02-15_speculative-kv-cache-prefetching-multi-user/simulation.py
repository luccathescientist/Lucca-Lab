import numpy as np
import matplotlib.pyplot as plt
import time

def simulate_prefetching():
    # Simulation parameters
    time_steps = 100
    cache_size = 1024  # MB
    user_requests = np.random.poisson(lam=5, size=time_steps)
    
    # Baseline: Demand-driven loading
    baseline_latency = []
    # Proposed: Speculative prefetching
    prefetch_latency = []
    
    current_cache = set()
    hit_rate_baseline = []
    hit_rate_prefetch = []

    for req in user_requests:
        # Baseline
        latency_b = 50 if req not in current_cache else 5
        baseline_latency.append(latency_b)
        
        # Prefetch (simulated 85% accuracy)
        is_prefetched = np.random.random() < 0.85
        latency_p = 5 if is_prefetched else 50
        prefetch_latency.append(latency_p)

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(baseline_latency, label='Baseline (Demand-Driven)', alpha=0.7)
    plt.plot(prefetch_latency, label='Speculative Prefetching', alpha=0.7)
    plt.title('Inference Latency: Baseline vs Speculative Prefetching (Simulation)')
    plt.xlabel('Request Index')
    plt.ylabel('Latency (ms)')
    plt.legend()
    plt.grid(True)
    plt.savefig('ml-explorations/2026-02-15_speculative-kv-cache-prefetching-multi-user/latency_comparison.png')
    
    avg_b = np.mean(baseline_latency)
    avg_p = np.mean(prefetch_latency)
    
    return avg_b, avg_p

if __name__ == "__main__":
    avg_b, avg_p = simulate_prefetching()
    print(f"Average Baseline Latency: {avg_b:.2f} ms")
    print(f"Average Prefetch Latency: {avg_p:.2f} ms")
    print(f"Improvement: {((avg_b - avg_p) / avg_b) * 100:.2f}%")
