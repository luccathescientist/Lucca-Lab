import numpy as np
import matplotlib.pyplot as plt
import time

def simulate_entropy_pruning(context_length, pruning_ratio=0.3):
    """
    Simulates token pruning based on attention entropy.
    Low entropy = high focus (keep), High entropy = uniform/unimportant (prune).
    """
    print(f"Starting simulation for context_length: {context_length}")
    
    # Simulate entropy scores (0 to 1)
    # In a real model, this would be -sum(p * log(p)) of the attention weights
    entropy_scores = np.random.beta(2, 5, context_length) 
    
    threshold = np.percentile(entropy_scores, (1 - pruning_ratio) * 100)
    
    # Pruning mask
    kept_indices = np.where(entropy_scores <= threshold)[0]
    pruned_indices = np.where(entropy_scores > threshold)[0]
    
    vram_initial = context_length * 0.125 # MB (approx for KV cache)
    vram_final = len(kept_indices) * 0.125
    
    return entropy_scores, kept_indices, pruned_indices, vram_initial, vram_final

def run_experiment():
    context_lengths = [32768, 65536, 131072, 262144, 524288, 1048576]
    results = []

    for cl in context_lengths:
        start = time.time()
        scores, kept, pruned, v_i, v_f = simulate_entropy_pruning(cl)
        duration = time.time() - start
        results.append({
            "context": cl,
            "kept": len(kept),
            "vram_saved": v_i - v_f,
            "latency_ms": duration * 1000
        })

    # Plotting
    contexts = [r['context'] for r in results]
    savings = [r['vram_saved'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(contexts, savings, marker='o', color='teal', linestyle='--')
    plt.title("VRAM Savings via Entropy-Driven Token Pruning (Simulation)")
    plt.xlabel("Context Length (Tokens)")
    plt.ylabel("VRAM Saved (MB)")
    plt.grid(True, alpha=0.3)
    plt.savefig("ml-explorations/2026-02-11_entropy-driven-token-pruning/vram_savings.png")
    
    return results

if __name__ == "__main__":
    res = run_experiment()
    for r in res:
        print(f"Context {r['context']}: Saved {r['vram_saved']:.2f} MB")
