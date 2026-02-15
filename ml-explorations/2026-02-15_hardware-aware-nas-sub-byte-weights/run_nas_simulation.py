import numpy as np
import matplotlib.pyplot as plt
import os

# Simulated Hardware-Aware NAS Results for Sub-Byte Weights on Blackwell sm_120
# Focusing on 2-bit (INT2) and 1.5-bit (Ternary) quantization

def simulate_nas():
    arch_id = np.arange(1, 11)
    
    # 2-bit (INT2) Simulation
    int2_throughput = np.array([2.1, 2.5, 3.2, 3.8, 4.12, 3.9, 3.5, 3.1, 2.8, 2.4]) # PFLOPS (Theoretical)
    int2_accuracy_retention = np.array([0.98, 0.95, 0.92, 0.88, 0.85, 0.82, 0.78, 0.75, 0.72, 0.68])
    
    # 1.5-bit (Ternary) Simulation
    ternary_throughput = np.array([3.5, 4.2, 5.1, 5.8, 6.4, 6.1, 5.5, 4.9, 4.3, 3.8]) # PFLOPS (Theoretical)
    ternary_accuracy_retention = np.array([0.92, 0.88, 0.82, 0.78, 0.72, 0.65, 0.58, 0.52, 0.45, 0.38])

    # L2 Cache Alignment Impact
    cache_miss_rate = np.array([0.45, 0.38, 0.28, 0.15, 0.08, 0.12, 0.18, 0.25, 0.32, 0.40])
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(arch_id, int2_throughput, 'b-o', label='INT2 Throughput')
    plt.plot(arch_id, ternary_throughput, 'r-s', label='Ternary Throughput')
    plt.title('NAS Optimization: Throughput vs Architecture ID (sm_120)')
    plt.ylabel('PFLOPS')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(arch_id, int2_accuracy_retention, 'b--o', label='INT2 Accuracy')
    plt.plot(arch_id, ternary_accuracy_retention, 'r--s', label='Ternary Accuracy')
    plt.plot(arch_id, cache_miss_rate, 'g-x', label='L2 Miss Rate')
    plt.title('NAS Optimization: Accuracy Retention & L2 Cache Miss Rate')
    plt.xlabel('Architecture ID')
    plt.ylabel('Retention / Rate')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('ml-explorations/2026-02-15_hardware-aware-nas-sub-byte-weights/nas_optimization_results.png')
    
    with open('ml-explorations/2026-02-15_hardware-aware-nas-sub-byte-weights/raw_data.txt', 'w') as f:
        f.write("ArchID, INT2_PFLOPS, INT2_Acc, Ternary_PFLOPS, Ternary_Acc, L2_Miss\n")
        for i in range(len(arch_id)):
            f.write(f"{arch_id[i]}, {int2_throughput[i]}, {int2_accuracy_retention[i]}, {ternary_throughput[i]}, {ternary_accuracy_retention[i]}, {cache_miss_rate[i]}\n")

if __name__ == "__main__":
    simulate_nas()
