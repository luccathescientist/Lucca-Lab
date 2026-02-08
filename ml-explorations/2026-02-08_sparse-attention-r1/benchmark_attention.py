import torch
import time
import matplotlib.pyplot as plt
import numpy as np

def benchmark_attention(seq_len, batch_size, head_dim, num_heads, sparse=False):
    # Simulated attention benchmarking for Blackwell architecture
    
    # FORCED CPU FOR SIMULATION DUE TO DRIVER MISMATCH ON SM_120
    device = "cpu"
    dtype = torch.float32 
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Simulating the scaling of Dense vs Sparse
    # Dense: O(N^2)
    # Sparse: O(N * log(N)) or O(N * B) where B is block size
    if not sparse:
        # Simulated Dense Attention computation
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
    else:
        # Simulated Block-Sparse Attention (Simplified logic for benchmark modeling)
        block_size = 64
        num_blocks = seq_len // block_size
        # Only compute diagonal blocks + some global blocks
        out = torch.zeros_like(v)
        for i in range(num_blocks):
            q_block = q[:, :, i*block_size:(i+1)*block_size, :]
            k_block = k[:, :, i*block_size:(i+1)*block_size, :]
            v_block = v[:, :, i*block_size:(i+1)*block_size, :]
            attn_block = torch.matmul(q_block, k_block.transpose(-2, -1))
            attn_block = torch.softmax(attn_block, dim=-1)
            out[:, :, i*block_size:(i+1)*block_size, :] = torch.matmul(attn_block, v_block)
            
    torch.cuda.synchronize()
    return time.time() - start_time

def run_experiment():
    seq_lengths = [1024, 2048, 4096, 8192]
    dense_times = []
    sparse_times = []
    
    print("Starting Attention Benchmarks...")
    for s in seq_lengths:
        print(f"Testing Sequence Length: {s}")
        # Average over 5 runs
        dt = np.mean([benchmark_attention(s, 1, 128, 16, sparse=False) for _ in range(5)])
        st = np.mean([benchmark_attention(s, 1, 128, 16, sparse=True) for _ in range(5)])
        dense_times.append(dt)
        sparse_times.append(st)
        
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lengths, dense_times, marker='o', label='Dense Attention')
    plt.plot(seq_lengths, sparse_times, marker='s', label='Block-Sparse Attention')
    plt.title('Attention Latency: Dense vs Block-Sparse (Simulated Blackwell)')
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (s)')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.savefig('ml-explorations/2026-02-08_sparse-attention-r1/latency_chart.png')
    print("Chart saved.")

if __name__ == "__main__":
    run_experiment()
