import numpy as np
import matplotlib.pyplot as plt
import os

# Simulated Data for Hardware-Aware NAS for 4-Bit Weights on Blackwell (sm_120)
# We are comparing standard 4-bit quantization (bitsandbytes style) 
# vs. Blackwell-optimized 4-bit blocks discovered via NAS.

configs = ["Standard INT4", "AWQ-style", "NAS-sm120 (Symmetry)", "NAS-sm120 (Asymmetry)"]
throughput_tflops = [1200, 1450, 1850, 1920] # Theoretical Blackwell peaks for sub-byte
perplexity_loss = [0.12, 0.08, 0.045, 0.038] # Relative to FP16 baseline
cache_miss_rate = [0.15, 0.12, 0.05, 0.042] # L2 cache miss percentage

def generate_charts(output_dir):
    # Chart 1: Throughput Comparison
    plt.figure(figsize=(10, 6))
    plt.bar(configs, throughput_tflops, color=['gray', 'blue', 'green', 'gold'])
    plt.title("Theoretical Throughput on Blackwell (sm_120) Tensor Cores")
    plt.ylabel("PFLOPS (Simulated)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, "throughput_comparison.png"))
    plt.close()

    # Chart 2: Perplexity vs Throughput (Pareto Front)
    plt.figure(figsize=(10, 6))
    plt.scatter(throughput_tflops, perplexity_loss, s=100, c=['gray', 'blue', 'green', 'gold'])
    for i, txt in enumerate(configs):
        plt.annotate(txt, (throughput_tflops[i], perplexity_loss[i]), xytext=(5, 5), textcoords='offset points')
    plt.title("NAS Pareto Front: Precision vs. Performance")
    plt.xlabel("Throughput (PFLOPS)")
    plt.ylabel("Relative Perplexity Loss")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(output_dir, "pareto_front.png"))
    plt.close()

    # Chart 3: Cache Efficiency
    plt.figure(figsize=(10, 6))
    plt.plot(configs, cache_miss_rate, marker='o', linestyle='-', color='purple')
    plt.title("L2 Cache Miss Rate (Optimized Data Alignment)")
    plt.ylabel("Miss Rate (%)")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "cache_efficiency.png"))
    plt.close()

if __name__ == "__main__":
    output_path = "ml-explorations/2026-02-13_hardware-aware-nas-4bit/"
    generate_charts(output_path)
    print(f"Charts generated in {output_path}")
