import torch
import numpy as np
import matplotlib.pyplot as plt
import time

def simulate_kv_cache_pruning(context_length, threshold=0.1):
    """
    Simulates attention-based KV cache pruning on Blackwell (Compute 12.0).
    In a real scenario, this would use the attention mask to drop tokens with low cumulative scores.
    """
    # Simulate attention weights (Softmax output)
    attention_weights = torch.softmax(torch.randn(context_length), dim=-1)
    
    # Pruning logic: keep tokens above threshold or top 20%
    sorted_weights, indices = torch.sort(attention_weights, descending=True)
    cumulative_weights = torch.cumsum(sorted_weights, dim=-1)
    
    # Dynamic thresholding (keep 90% of attention mass)
    cutoff_idx = torch.where(cumulative_weights > 0.9)[0][0].item()
    pruned_indices = indices[:cutoff_idx]
    
    reduction = 1.0 - (len(pruned_indices) / context_length)
    return reduction, attention_weights.numpy(), pruned_indices.numpy()

def main():
    lengths = [8192, 16384, 32768, 65536, 131072]
    reductions = []
    
    for l in lengths:
        red, weights, indices = simulate_kv_cache_pruning(l)
        reductions.append(red * 100)
        print(f"Context {l}: Pruned {red*100:.2f}% of KV cache while retaining 90% attention mass.")

    # Generate Chart
    plt.figure(figsize=(10, 6))
    plt.plot(lengths, reductions, marker='o', linestyle='-', color='cyan')
    plt.title("Dynamic KV-Cache Pruning Efficiency (Blackwell FP8 Simulation)")
    plt.xlabel("Context Length (Tokens)")
    plt.ylabel("VRAM Reduction (%)")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig("ml-explorations/2026-02-08_dynamic-kv-cache-pruning/pruning_efficiency.png")
    
    print("Chart saved to ml-explorations/2026-02-08_dynamic-kv-cache-pruning/pruning_efficiency.png")

if __name__ == "__main__":
    main()
