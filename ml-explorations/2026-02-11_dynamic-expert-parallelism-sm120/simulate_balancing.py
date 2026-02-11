import numpy as np
import matplotlib.pyplot as plt

def simulate_load_balancing():
    # Simulate activation density for a 16-expert MoE
    experts = np.arange(16)
    activation_density = np.random.gamma(shape=2, scale=1, size=16)
    
    # Baseline: Static Assignment (Uniform distribution across 4 TPCs)
    static_load = np.array_split(activation_density, 4)
    static_sums = [np.sum(s) for s in static_load]
    
    # Proposed: Dynamic Load Balancing (Greedy bin packing)
    sorted_indices = np.argsort(activation_density)[::-1]
    dynamic_tpc_loads = [0.0] * 4
    dynamic_assignments = [[] for _ in range(4)]
    
    for idx in sorted_indices:
        min_tpc = np.argmin(dynamic_tpc_loads)
        dynamic_tpc_loads[min_tpc] += activation_density[idx]
        dynamic_assignments[min_tpc].append(idx)
        
    # Stats
    static_std = np.std(static_sums)
    dynamic_std = np.std(dynamic_tpc_loads)
    improvement = (static_std - dynamic_std) / static_std * 100
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.bar(range(4), static_sums, color='skyblue', label='Static')
    ax1.set_title(f'Static Assignment (Std: {static_std:.2f})')
    ax1.set_ylabel('Total Activation Density')
    ax1.set_xlabel('TPC ID')
    
    ax2.bar(range(4), dynamic_tpc_loads, color='salmon', label='Dynamic')
    ax2.set_title(f'Dynamic Balancing (Std: {dynamic_std:.2f})')
    ax2.set_xlabel('TPC ID')
    
    plt.tight_layout()
    plt.savefig('ml-explorations/2026-02-11_dynamic-expert-parallelism-sm120/load_balance_comparison.png')
    
    return improvement, static_sums, dynamic_tpc_loads

if __name__ == "__main__":
    imp, s_sums, d_sums = simulate_load_balancing()
    print(f"Improvement in Load Balance (Std Dev reduction): {imp:.2f}%")
    print(f"Static TPC Loads: {s_sums}")
    print(f"Dynamic TPC Loads: {d_sums}")
