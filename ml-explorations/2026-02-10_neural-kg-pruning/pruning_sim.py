import numpy as np
import time
import matplotlib.pyplot as plt
import os

def simulate_kg_pruning():
    # Simulation parameters
    node_counts = np.arange(1000, 50001, 5000)
    latencies_no_pruning = node_counts * 0.00005 + np.random.normal(0, 0.001, len(node_counts))
    
    # Pruning logic: keep the graph size capped at 10k active nodes
    pruned_node_counts = np.array([min(n, 10000) for n in node_counts])
    latencies_with_pruning = pruned_node_counts * 0.00005 + np.random.normal(0, 0.0005, len(node_counts))
    
    # Add base overhead for the pruning check
    latencies_with_pruning += 0.005 

    plt.figure(figsize=(10, 6))
    plt.plot(node_counts, latencies_no_pruning, 'r--', label='Without Pruning (Linear Growth)')
    plt.plot(node_counts, latencies_with_pruning, 'g-', label='With Semantic Decay Pruning (Capped)')
    plt.title('Knowledge Graph Retrieval Latency vs. Node Count')
    plt.xlabel('Total Knowledge Nodes')
    plt.ylabel('Latency (seconds)')
    plt.legend()
    plt.grid(True)
    
    output_path = 'ml-explorations/2026-02-10_neural-kg-pruning/latency_chart.png'
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")

    # Log results for REPORT.md
    return node_counts, latencies_no_pruning, latencies_with_pruning

if __name__ == "__main__":
    simulate_kg_pruning()
