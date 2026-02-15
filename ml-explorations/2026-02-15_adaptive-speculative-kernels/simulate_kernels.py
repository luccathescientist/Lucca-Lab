import numpy as np
import time
import matplotlib.pyplot as plt
import os

def simulate_kernel_dispatch(num_iterations=1000, batch_size=128):
    """
    Simulates adaptive kernel dispatch for Blackwell sm_120.
    Toggles between FP8 (high precision) and INT4 (high throughput) based on 
    simulated tensor distribution.
    """
    # Performance metrics (hypothetical PFLOPS for sm_120)
    FP8_PFLOPS = 1.2
    INT4_PFLOPS = 2.4
    DISPATCH_OVERHEAD_MS = 0.045 # 45 microseconds
    
    # Kernel compute times (ms)
    compute_fp8 = (batch_size * 4096 * 4096) / (FP8_PFLOPS * 1e15) * 1000
    compute_int4 = (batch_size * 4096 * 4096) / (INT4_PFLOPS * 1e15) * 1000
    
    results = []
    
    for i in range(num_iterations):
        # Adaptive logic: swap if precision requirement is low (random for simulation)
        precision_req = np.random.rand()
        
        start_time = time.time()
        
        if precision_req > 0.3: # 70% chance to use INT4 (hypothetically)
            # Dispatch INT4
            actual_compute = compute_int4 + DISPATCH_OVERHEAD_MS
            kernel_type = "INT4"
        else:
            # Dispatch FP8
            actual_compute = compute_fp8 + DISPATCH_OVERHEAD_MS
            kernel_type = "FP8"
            
        end_time = time.time()
        
        results.append({
            "iteration": i,
            "kernel": kernel_type,
            "latency_ms": actual_compute,
            "precision_score": precision_req
        })
        
    return results

def plot_results(results, output_path):
    latencies = [r['latency_ms'] for r in results]
    kernels = [r['kernel'] for r in results]
    
    plt.figure(figsize=(12, 6))
    
    # Scatter plot
    colors = ['blue' if k == 'FP8' else 'green' for k in kernels]
    plt.scatter(range(len(latencies)), latencies, c=colors, alpha=0.5, s=10)
    
    plt.title("Adaptive Speculative Kernel Latency (Simulated sm_120)")
    plt.xlabel("Iteration")
    plt.ylabel("Latency (ms)")
    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', label='FP8 Kernel', markerfacecolor='blue', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='INT4 Kernel', markerfacecolor='green', markersize=10)
    ])
    
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    print("Running Blackwell Adaptive Kernel Simulation...")
    sim_results = simulate_kernel_dispatch()
    
    output_dir = "ml-explorations/2026-02-15_adaptive-speculative-kernels"
    os.makedirs(output_dir, exist_ok=True)
    
    plot_results(sim_results, os.path.join(output_dir, "latency_dispatch.png"))
    
    # Calculate stats
    avg_latency = np.mean([r['latency_ms'] for r in sim_results])
    fp8_count = len([r for r in sim_results if r['kernel'] == 'FP8'])
    int4_count = len([r for r in sim_results if r['kernel'] == 'INT4'])
    
    with open(os.path.join(output_dir, "stats.txt"), "w") as f:
        f.write(f"Average Latency: {avg_latency:.4f} ms\n")
        f.write(f"FP8 Kernels: {fp8_count}\n")
        f.write(f"INT4 Kernels: {int4_count}\n")
        f.write(f"Throughput Gain (vs Static FP8): {( ( (4096*4096*128)/(1.2e15) * 1000 ) / avg_latency ):.2f}x\n")

    print(f"Simulation complete. Results saved to {output_dir}")
