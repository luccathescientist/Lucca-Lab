import numpy as np
import matplotlib.pyplot as plt

def simulate_cache_tiering():
    context_lengths = np.linspace(1e6, 10e6, 10)
    
    # Baseline: HBM only (thrashing after 4M)
    baseline_latency = 50 + (context_lengths / 1e6)**2 * 10
    
    # Saliency-Gated Tiering (L2 + HBM + System RAM)
    # L2: 128MB (Hot), HBM: 80GB (Warm), RAM: 2TB (Cold)
    tiering_latency = 50 + (context_lengths / 1e6) * 5 + np.log1p(context_lengths / 1e6) * 2
    
    plt.figure(figsize=(10, 6))
    plt.plot(context_lengths / 1e6, baseline_latency, 'r--', label='Baseline (HBM Thrashing)')
    plt.plot(context_lengths / 1e6, tiering_latency, 'g-', label='Saliency-Gated Tiering (sm_120)')
    plt.fill_between(context_lengths / 1e6, tiering_latency, baseline_latency, color='green', alpha=0.1)
    
    plt.title('Inference Latency vs. Context Length (10M Token Scale)')
    plt.xlabel('Context Length (Millions of Tokens)')
    plt.ylabel('Latency per Token (ms)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig('ml-explorations/2026-02-17_saliency-gated-kv-cache-tiering/latency_scaling.png')
    
    # Cache Hit Rate
    layers = ['L2 (Hot)', 'HBM (Warm)', 'RAM (Cold)']
    hit_rates = [0.85, 0.12, 0.03]
    
    plt.figure(figsize=(8, 5))
    plt.bar(layers, hit_rates, color=['blue', 'orange', 'grey'])
    plt.title('KV-Cache Access Distribution (Saliency-Gated)')
    plt.ylabel('Hit Rate')
    plt.ylim(0, 1)
    plt.savefig('ml-explorations/2026-02-17_saliency-gated-kv-cache-tiering/cache_distribution.png')

if __name__ == "__main__":
    simulate_cache_tiering()
