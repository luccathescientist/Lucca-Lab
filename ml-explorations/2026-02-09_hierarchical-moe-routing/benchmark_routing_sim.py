import time
import matplotlib.pyplot as plt
import numpy as np

# Configuration
NUM_EXPERTS = 64
NUM_CLUSTERS = 8
EXPERTS_PER_CLUSTER = 8
D_MODEL = 1024
TOP_K_CLUSTERS = 1
TOP_K_EXPERTS_PER_CLUSTER = 2
ITERATIONS = 1000

def simulate_routing():
    print("Simulating Hierarchical MoE Routing Latency (CPU Simulation)...")
    
    # Simulate Flat Routing (Single Matrix Multiply + Top-K)
    flat_times = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        # Simulated D_MODEL -> NUM_EXPERTS projection
        _ = np.random.randn(1, D_MODEL) @ np.random.randn(D_MODEL, NUM_EXPERTS)
        # Simulated Top-K
        _ = np.argsort(np.random.randn(NUM_EXPERTS))[-2:]
        flat_times.append(time.perf_counter() - start)
        
    # Simulate Hierarchical Routing (Two smaller projections + Two Top-Ks)
    h_times = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        # Stage 1: D_MODEL -> NUM_CLUSTERS
        _ = np.random.randn(1, D_MODEL) @ np.random.randn(D_MODEL, NUM_CLUSTERS)
        _ = np.argsort(np.random.randn(NUM_CLUSTERS))[-1:]
        
        # Stage 2: D_MODEL -> EXPERTS_PER_CLUSTER (Specialist Router)
        _ = np.random.randn(1, D_MODEL) @ np.random.randn(D_MODEL, EXPERTS_PER_CLUSTER)
        _ = np.argsort(np.random.randn(EXPERTS_PER_CLUSTER))[-2:]
        h_times.append(time.perf_counter() - start)
        
    avg_flat = np.mean(flat_times) * 1000
    avg_h = np.mean(h_times) * 1000
    
    print(f"Average Flat Routing: {avg_flat:.4f} ms")
    print(f"Average Hierarchical Routing: {avg_h:.4f} ms")
    print(f"Speedup/Overhead: {avg_flat/avg_h:.2f}x")
    
    # Visualization
    labels = ['Flat (64 Experts)', 'Hierarchical (8x8 Clusters)']
    times = [avg_flat, avg_h]
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, times, color=['#00cfcf', '#ff00ff'])
    plt.ylabel('Latency (ms)')
    plt.title('MoE Routing Latency Simulation: Flat vs Hierarchical')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('routing_benchmark.png')
    print("Chart saved as routing_benchmark.png")

if __name__ == "__main__":
    simulate_routing()
