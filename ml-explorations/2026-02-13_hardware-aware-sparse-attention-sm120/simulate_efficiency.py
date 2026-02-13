import matplotlib.pyplot as plt
import time
import numpy as np

def simulate_sparse_attention_efficiency(context_length, l2_cache_size_mb=60):
    """
    Simulates the efficiency of sparse attention patterns relative to Blackwell L2 cache boundaries.
    Blackwell sm_120 (RTX 6000) has roughly 60MB of L2 cache.
    """
    # Define tile sizes that align with L2 (hypothetically)
    # Each FP8 element is 1 byte.
    # Standard tile for FlashAttention might be 128x128 = 16k elements = 16KB.
    # For a context of 128k, full attention is 128k * 128k = 16B elements = 16GB (OOM).
    
    tile_size = 128
    l2_capacity_elements = (l2_cache_size_mb * 1024 * 1024) # 1 element = 1 byte (FP8)
    
    # Sparse pattern: Block-Diagonal with alignment to L2 segments
    # Assume L2 is split into 128 segments of ~0.5MB each for parallelism.
    segment_size = l2_capacity_elements // 128
    
    # Efficiency calculation: (Managed Cache Hits) / (Total Accesses)
    # Dense: Total Accesses = L^2
    # Sparse: Total Accesses = L * (Block_Size)
    
    context_lengths = np.linspace(1024, 128000, 20, dtype=int)
    dense_miss_rate = []
    sparse_miss_rate = []
    
    for l in context_lengths:
        # Full dense attention complexity
        total_ops = l * l
        # Theoretical cache misses if L2 is exceeded
        dense_misses = max(0, total_ops - l2_capacity_elements) / total_ops
        dense_miss_rate.append(dense_misses)
        
        # Hardware-aware sparse (sliding window + global anchors)
        # aligned to L2 segment boundaries
        window_size = 2048 # local window
        global_anchors = 128
        sparse_ops = l * (window_size + global_anchors)
        # Sparse fits mostly in L2
        sparse_misses = max(0, sparse_ops - l2_capacity_elements) / sparse_ops
        sparse_miss_rate.append(sparse_misses)

    plt.figure(figsize=(10, 6))
    plt.plot(context_lengths, dense_miss_rate, label='Dense Attention Miss Rate (Simulated)', color='red')
    plt.plot(context_lengths, sparse_miss_rate, label='Hardware-Aware Sparse Miss Rate (Simulated)', color='green')
    plt.axvline(x=65536, linestyle='--', color='gray', label='64k Context Threshold')
    plt.title('Cache Efficiency vs Context Length (Blackwell sm_120)')
    plt.xlabel('Context Length (Tokens)')
    plt.ylabel('Cache Miss Probability')
    plt.legend()
    plt.grid(True)
    plt.savefig('cache_efficiency.png')
    
    print(f"Simulation complete. Efficiency gain at 128k: {(dense_miss_rate[-1]/max(0.001, sparse_miss_rate[-1])):.2f}x")

if __name__ == "__main__":
    simulate_sparse_attention_efficiency(128000)
