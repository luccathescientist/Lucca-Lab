import time
import json
import os
import matplotlib.pyplot as plt

# Simulated Blackwell sm_120 kernel profiling
def profile_kernel(block_size, threads_per_block):
    # This simulates a CUDA kernel measurement on Blackwell
    # Reward is 1 / latency (higher is better)
    # We add some synthetic logic: Blackwell prefers multiples of 32 (warps) 
    # and specific block sizes for its L1/Shared Memory hierarchy
    
    # Base latency in microseconds
    base_latency = 50.0 
    
    # Penalize non-warp alignments
    warp_penalty = 0 if threads_per_block % 32 == 0 else 20.0
    
    # Optimization "sweet spot" for Blackwell (hypothetically 256 or 512)
    sweet_spot_penalty = abs(threads_per_block - 256) * 0.05
    
    # Block size interaction
    block_penalty = abs(block_size - 128) * 0.1
    
    latency = base_latency + warp_penalty + sweet_spot_penalty + block_penalty
    return max(5.0, latency)

def run_rl_search():
    results = []
    best_latency = float('inf')
    best_config = None
    
    print("Starting SSKO (Self-Supervised Kernel Optimization) Search...")
    
    # Simple Grid Search as a proxy for the RL explorer's initial iterations
    for b_size in [64, 128, 256]:
        for t_size in [32, 64, 128, 256, 512, 1024]:
            latency = profile_kernel(b_size, t_size)
            results.append({"block_size": b_size, "threads": t_size, "latency": latency})
            
            if latency < best_latency:
                best_latency = latency
                best_config = (b_size, t_size)
                
    # Save results
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Generate Chart
    threads = [r["threads"] for r in results if r["block_size"] == 128]
    latencies = [r["latency"] for r in results if r["block_size"] == 128]
    
    plt.figure(figsize=(10, 6))
    plt.plot(threads, latencies, marker='o', linestyle='-', color='cyan')
    plt.title("SSKO Latency Optimization (Block Size 128) on Blackwell sm_120")
    plt.xlabel("Threads Per Block")
    plt.ylabel("Latency (μs)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("latency_chart.png")
    
    return best_config, best_latency

if __name__ == "__main__":
    best_cfg, best_lat = run_rl_search()
    print(f"Optimization Complete. Best Config: {best_cfg} with Latency: {best_lat:.2f}μs")
