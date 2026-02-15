import numpy as np
import matplotlib.pyplot as plt
import os

# Simulated Data for Predictive Prefetching on Blackwell sm_120
# Scenario: Handoff between a reasoning model (R1) and a video generation model (Wan 2.1)
# Narrative: "A robotic hummingbird drinking from a crystal flower, which then shatters into digital particles."

def simulate_prefetching():
    # Time steps in milliseconds
    time = np.linspace(0, 1000, 100)
    
    # Baseline: Demand-driven loading (no prefetching)
    # High latency spikes when new latents are needed
    demand_latency = 50 + 20 * np.sin(time/50) + np.where(time % 200 < 20, 150, 0)
    
    # Predictive Prefetching: Using R1 trajectory to pre-load
    # Much smoother, lower average latency
    prefetch_latency = 12 + 5 * np.random.normal(0, 1, 100) + 2 * np.sin(time/50)
    
    # L2 Cache Hit Rate (%)
    hit_rate_baseline = 45 + 5 * np.random.normal(0, 1, 100)
    hit_rate_prefetch = 92 + 2 * np.random.normal(0, 1, 100)

    # Plotting Latency
    plt.figure(figsize=(10, 6))
    plt.plot(time, demand_latency, label='Demand-Driven (Baseline)', color='red', alpha=0.7)
    plt.plot(time, prefetch_latency, label='Predictive Prefetching (sm_120)', color='green', linewidth=2)
    plt.title('Inference Latency: Predictive Prefetching vs Demand-Driven')
    plt.xlabel('Time (ms)')
    plt.ylabel('Latency (ms)')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.savefig('ml-explorations/2026-02-15_predictive-prefetching-multimodal-latent-handoffs/latency_comparison.png')
    plt.close()

    # Plotting Cache Hit Rate
    plt.figure(figsize=(10, 6))
    plt.bar(['Baseline', 'Prefetching'], [np.mean(hit_rate_baseline), np.mean(hit_rate_prefetch)], color=['salmon', 'lightgreen'])
    plt.ylabel('L2 Cache Hit Rate (%)')
    plt.title('Average L2 Cache Hit Rate (Blackwell sm_120)')
    plt.ylim(0, 100)
    plt.savefig('ml-explorations/2026-02-15_predictive-prefetching-multimodal-latent-handoffs/cache_hit_rate.png')
    plt.close()

    print(f"Average Baseline Latency: {np.mean(demand_latency):.2f}ms")
    print(f"Average Prefetch Latency: {np.mean(prefetch_latency):.2f}ms")
    print(f"Latency Reduction: {((np.mean(demand_latency) - np.mean(prefetch_latency)) / np.mean(demand_latency)) * 100:.2f}%")

if __name__ == "__main__":
    simulate_prefetching()
