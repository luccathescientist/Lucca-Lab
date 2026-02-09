import torch
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import os

def simulate_kv_cache_compaction(seq_len, compression_ratio_filler, compression_ratio_logic, logic_token_ratio=0.1):
    """
    Simulates adaptive KV-cache compaction.
    Logic tokens (key reasoning steps) are kept at higher precision/density.
    Filler tokens (articles, common words) are compressed more aggressively.
    """
    # Simulate semantic importance scores
    scores = torch.rand(seq_len)
    threshold = 1 - logic_token_ratio
    is_logic = scores > threshold
    
    # Simulate memory footprint
    # Base memory per token (normalized)
    base_mem = 1.0
    
    compaction_factors = torch.where(is_logic, 
                                     torch.tensor(1 - compression_ratio_logic), 
                                     torch.tensor(1 - compression_ratio_filler))
    
    compacted_mem = (base_mem * compaction_factors).sum().item()
    original_mem = float(seq_len)
    
    reduction = (1 - (compacted_mem / original_mem)) * 100
    return reduction, is_logic

def run_benchmark():
    seq_lens = [1024, 4096, 16384, 32768, 65536]
    filler_ratios = [0.2, 0.4, 0.6, 0.8]
    logic_ratio = 0.05 # Keep 95% of logic fidelity
    
    results = {}
    
    for f_ratio in filler_ratios:
        reductions = []
        for s_len in seq_lens:
            red, _ = simulate_kv_cache_compaction(s_len, f_ratio, logic_ratio)
            reductions.append(red)
        results[f"Filler Compaction {int(f_ratio*100)}%"] = reductions

    # Plotting
    plt.figure(figsize=(10, 6))
    for label, data in results.items():
        plt.plot(seq_lens, data, marker='o', label=label)
    
    plt.title("Adaptive KV-Cache Compaction Efficiency (Simulated)")
    plt.xlabel("Sequence Length")
    plt.ylabel("VRAM Reduction (%)")
    plt.xscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.savefig("compaction_efficiency.png")
    
    # Save raw data
    with open("results.txt", "w") as f:
        f.write(str(results))

if __name__ == "__main__":
    print("Starting Adaptive KV-Cache Compaction Simulation...")
    start = time.time()
    run_benchmark()
    end = time.time()
    print(f"Simulation completed in {end-start:.2f} seconds.")
