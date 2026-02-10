import numpy as np
import matplotlib.pyplot as plt
import os

# Simulated Entropy-Gated Progressive Quantization (NumPy Version)
# Task: Dynamically switch precision based on attention head entropy.

def calculate_entropy(probs):
    """Calculate Shannon entropy of attention probabilities."""
    return -np.sum(probs * np.log(probs + 1e-9), axis=-1)

def simulate_attention_head(batch_size, seq_len, num_heads):
    # Simulate some structured (low entropy) and some noisy (high entropy) attention
    heads = []
    for h in range(num_heads):
        if h < num_heads // 2:
            # Low entropy: one or two high peaks
            dist = np.zeros((batch_size, seq_len))
            idx = np.random.randint(0, seq_len, size=batch_size)
            for i in range(batch_size):
                dist[i, idx[i]] = 0.9
                dist[i] += np.random.dirichlet([0.1]*seq_len) * 0.1
        else:
            # High entropy: uniform/noisy
            dist = np.random.dirichlet([10.0]*seq_len, size=batch_size)
        heads.append(dist)
    
    return np.stack(heads, axis=1) # [batch, heads, seq]

def gate_precision(entropy, thresholds):
    """
    thresholds: [t_int4, t_fp8]
    """
    precisions = np.zeros_like(entropy, dtype=int)
    precisions[entropy < thresholds[0]] = 0 # INT4
    precisions[(entropy >= thresholds[0]) & (entropy < thresholds[1])] = 1 # FP8
    precisions[entropy >= thresholds[1]] = 2 # FP16
    return precisions

def run_simulation():
    batch_size = 4
    seq_len = 128
    num_heads = 32
    
    # Thresholds for entropy gating
    thresholds = [1.5, 3.5]
    
    attn_probs = simulate_attention_head(batch_size, seq_len, num_heads)
    entropy = calculate_entropy(attn_probs)
    
    precisions = gate_precision(entropy, thresholds)
    
    # Calculate stats
    int4_count = np.sum(precisions == 0)
    fp8_count = np.sum(precisions == 1)
    fp16_count = np.sum(precisions == 2)
    total = precisions.size
    
    print(f"Simulation Results:")
    print(f"Total Attention Blocks: {total}")
    print(f"INT4 (Low Entropy/High Focus): {int4_count} ({int4_count/total:.2%})")
    print(f"FP8 (Medium Entropy/Contextual): {fp8_count} ({fp8_count/total:.2%})")
    print(f"FP16 (High Entropy/Noisy/Critical): {fp16_count} ({fp16_count/total:.2%})")
    
    # Estimated throughput speedup (Blackwell theoretical: FP16=1, FP8=2, INT4=4)
    speedup = (int4_count * 4 + fp8_count * 2 + fp16_count * 1) / total
    print(f"Theoretical Speedup over FP16: {speedup:.2f}x")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.hist(entropy.flatten(), bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(thresholds[0], color='green', linestyle='--', label='INT4 Threshold')
    plt.axvline(thresholds[1], color='orange', linestyle='--', label='FP8 Threshold')
    plt.title('Entropy Distribution of Attention Heads (Simulated)')
    plt.xlabel('Entropy')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('plots/entropy_distribution.png')
    
    plt.figure(figsize=(8, 8))
    labels = ['INT4', 'FP8', 'FP16']
    sizes = [int4_count, fp8_count, fp16_count]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title('Precision Distribution Across Attention Layers')
    plt.savefig('plots/precision_pie.png')

if __name__ == "__main__":
    run_simulation()
