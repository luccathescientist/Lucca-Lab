import numpy as np
import matplotlib.pyplot as plt
import time

# Simulation Parameters
L2_CACHE_SIZE_MB = 128  # RTX 6000 Blackwell L2 Cache
KV_CACHE_PER_TOKEN_MB = 0.004  # Rough estimate for a mid-sized model (e.g., 32B FP8)
MAX_PREFETCH_TOKENS = int(L2_CACHE_SIZE_MB * 0.7 / KV_CACHE_PER_TOKEN_MB) # Use 70% of L2
USER_SESSIONS = 10
TIME_STEPS = 100

def simulate_prefetching():
    """
    Simulates the performance gain of speculative KV-cache prefetching.
    """
    # Probability of a user request occurring (0 to 1)
    request_probability = np.random.uniform(0.1, 0.4, size=USER_SESSIONS)
    
    # Latency without prefetching (ms)
    base_latency = 15.0 
    # Latency with prefetching (ms) - hits L2 instead of VRAM
    prefetch_latency = 1.2
    
    # Trackers
    latencies_standard = []
    latencies_prefetch = []
    hit_rates = []

    for t in range(TIME_STEPS):
        # Actual requests this step
        requests = np.random.rand(USER_SESSIONS) < request_probability
        
        # Speculative Prefetcher (Predicts requests with some accuracy)
        prediction_accuracy = 0.85
        predictions = (np.random.rand(USER_SESSIONS) < (request_probability * 1.2)) & (np.random.rand(USER_SESSIONS) < prediction_accuracy)
        
        # Calculate Latency
        step_latency_std = np.sum(requests * base_latency)
        
        # Hit/Miss Logic
        hits = requests & predictions
        misses = requests & ~predictions
        step_latency_pref = np.sum(hits * prefetch_latency) + np.sum(misses * base_latency)
        
        latencies_standard.append(step_latency_std)
        latencies_prefetch.append(step_latency_pref)
        
        if np.sum(requests) > 0:
            hit_rates.append(np.sum(hits) / np.sum(requests))

    # Calculate Totals
    total_reduction = (1 - np.sum(latencies_prefetch) / np.sum(latencies_standard)) * 100
    avg_hit_rate = np.mean(hit_rates) * 100

    # Generate Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(latencies_standard, label='Standard Request (VRAM)', alpha=0.7)
    plt.plot(latencies_prefetch, label='Prefetched Request (L2-Backed)', color='orange')
    plt.title(f'Speculative KV-Cache Prefetching Performance (sm_120)\nTotal Latency Reduction: {total_reduction:.2f}% | Hit Rate: {avg_hit_rate:.1f}%')
    plt.xlabel('Time Steps')
    plt.ylabel('Aggregate Latency (ms)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('ml-explorations/2026-02-14_speculative-kv-cache-prefetching/performance_chart.png')
    
    return total_reduction, avg_hit_rate

if __name__ == "__main__":
    print("Starting KV-Cache Prefetching Simulation...")
    reduction, hit_rate = simulate_prefetching()
    print(f"Simulation Complete. Reduction: {reduction:.2f}%, Hit Rate: {hit_rate:.2f}%")
