import time
import matplotlib.pyplot as plt
import numpy as np

# Simulation of Hardware-Aware Sparse Attention for Blackwell sm_120
# Focusing on L2 cache alignment (512KB segments)

class BlackwellSparseAttention:
    def __init__(self, seq_len, head_dim, window_size=2048, l2_segment_size=512 * 1024):
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.window_size = window_size
        self.l2_segment_size = l2_segment_size
        
        # Calculate how many tokens fit into an L2 segment
        # FP16 (2 bytes) * head_dim
        self.tokens_per_l2 = l2_segment_size // (2 * head_dim)
        
    def simulate_attention(self, mode='dense'):
        start_time = time.time()
        # Mocking compute and memory access patterns
        if mode == 'dense':
            # Dense attention O(N^2)
            ops = self.seq_len ** 2 * self.head_dim
            # Simulate latency based on complexity - scaled for simulation speed
            delay = (ops / 1e12) * 10 # Scaling for visible difference
            time.sleep(min(delay, 0.5)) 
        elif mode == 'sparse_l2_aligned':
            # Sparse attention O(N * window_size)
            # Aligned to L2 segments to reduce cache misses
            ops = self.seq_len * self.window_size * self.head_dim
            # Efficiency gain from L2 hits (simulated)
            delay = (ops / (1e12 * 1.5)) * 10
            time.sleep(min(delay, 0.5))
            
        return time.time() - start_time

def run_simulation():
    seq_lengths = [32768, 65536, 131072, 262144, 524288]
    dense_latencies = []
    sparse_latencies = []
    cache_miss_rates_dense = []
    cache_miss_rates_sparse = []

    for n in seq_lengths:
        model = BlackwellSparseAttention(seq_len=n, head_dim=128)
        
        # Simulating dense
        d_lat = model.simulate_attention('dense')
        dense_latencies.append(d_lat * 1000) # ms
        cache_miss_rates_dense.append(45.0 + (n / 1000000) * 10) # Mock trend
        
        # Simulating sparse L2-aligned
        s_lat = model.simulate_attention('sparse_l2_aligned')
        sparse_latencies.append(s_lat * 1000) # ms
        cache_miss_rates_sparse.append(8.0 + (n / 1000000) * 2) # Mock trend

    # Plotting
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(seq_lengths, dense_latencies, label='Dense Attention', marker='o')
    plt.plot(seq_lengths, sparse_latencies, label='L2-Aligned Sparse', marker='s')
    plt.title('Inference Latency vs Context Length')
    plt.xlabel('Sequence Length')
    plt.ylabel('Latency (ms)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(seq_lengths, cache_miss_rates_dense, label='Dense Cache Miss %', marker='o', color='red')
    plt.plot(seq_lengths, cache_miss_rates_sparse, label='Sparse Cache Miss %', marker='s', color='green')
    plt.title('Simulated L2 Cache Miss Rate')
    plt.xlabel('Sequence Length')
    plt.ylabel('Miss Rate (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('latency_cache_analysis.png')
    
    print(f"Simulation complete. Results saved to latency_cache_analysis.png")
    print(f"Final Seq Length: {seq_lengths[-1]}")
    print(f"Dense Latency: {dense_latencies[-1]:.2f}ms")
    print(f"Sparse Latency: {sparse_latencies[-1]:.2f}ms")
    print(f"Cache Miss Reduction: {cache_miss_rates_dense[-1] - cache_miss_rates_sparse[-1]:.2f}%")

if __name__ == "__main__":
    run_simulation()
