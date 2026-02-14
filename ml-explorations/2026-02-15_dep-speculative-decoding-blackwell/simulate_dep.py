import time
import matplotlib.pyplot as plt
import numpy as np

def simulate_blackwell_throughput(batch_size, seq_len, model_size_b, precision='FP8', use_dep=False, speculative_factor=1.0):
    """
    Simulates Blackwell throughput for DEP + Speculative Decoding.
    Parameters:
        model_size_b: Model size in Billion parameters.
        precision: 'FP8' or 'INT4'.
        use_dep: Data+Expert Parallel.
        speculative_factor: Throughput multiplier from speculative decoding.
    """
    # Theoretical Blackwell Peak for sm_120 (RTX 6000 style)
    # 1.8 PFLOPS FP8, 3.6 PFLOPS INT4 (scaled)
    peak_flops = 1.8e15 if precision == 'FP8' else 3.6e15
    
    # Simple roofline-ish model
    # FLOPs per token = 2 * params
    flops_per_token = 2 * model_size_b * 1e9
    
    # Efficiency factors
    base_efficiency = 0.45 # Standard vLLM/TRT-LLM efficiency
    dep_boost = 1.35 if use_dep else 1.0
    
    tokens_per_sec = (peak_flops * base_efficiency * dep_boost * speculative_factor) / flops_per_token
    # Scale by batch size (simplified)
    tokens_per_sec = tokens_per_sec * (batch_size / 32) # Normalized to batch 32
    
    return tokens_per_sec

def run_experiment():
    batch_sizes = [8, 16, 32, 64, 128]
    model_size = 120 # 120B model
    
    baseline = [simulate_blackwell_throughput(b, 1024, model_size, 'FP8', False, 1.0) for b in batch_sizes]
    dep_only = [simulate_blackwell_throughput(b, 1024, model_size, 'FP8', True, 1.0) for b in batch_sizes]
    dep_speculative = [simulate_blackwell_throughput(b, 1024, model_size, 'FP8', True, 2.5) for b in batch_sizes]
    
    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, baseline, marker='o', label='Baseline (FP8)')
    plt.plot(batch_sizes, dep_only, marker='s', label='DEP Optimized')
    plt.plot(batch_sizes, dep_speculative, marker='^', label='DEP + Speculative (Eagle)')
    
    plt.title(f'Blackwell sm_120 Throughput Simulation ({model_size}B Model)')
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (Tokens/Sec)')
    plt.legend()
    plt.grid(True)
    plt.savefig('ml-explorations/2026-02-15_dep-speculative-decoding-blackwell/throughput_chart.png')
    
    print(f"Final Throughput (Batch 128, DEP+Spec): {dep_speculative[-1]:.2f} tokens/sec")
    
    with open('ml-explorations/2026-02-15_dep-speculative-decoding-blackwell/raw_results.txt', 'w') as f:
        f.write(f"Batch Sizes: {batch_sizes}\n")
        f.write(f"Baseline: {baseline}\n")
        f.write(f"DEP+Spec: {dep_speculative}\n")

if __name__ == "__main__":
    run_experiment()
