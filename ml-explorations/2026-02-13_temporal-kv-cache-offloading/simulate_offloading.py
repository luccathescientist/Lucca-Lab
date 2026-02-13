import time
import numpy as np
import matplotlib.pyplot as plt
import os

def simulate_offloading():
    # Simulation parameters for RTX 6000 Blackwell (sm_120)
    # PCIe Gen5 x16 theoretical: ~64 GB/s (one way)
    # NVMe peak (Gen5): ~14 GB/s
    
    # We'll simulate offloading 8GB of KV-cache (roughly 1M tokens in FP16 for a medium model)
    cache_size_gb = 8.0
    pcie_bandwidth_gbps = 58.0  # Real-world sustained
    nvme_bandwidth_gbps = 12.0  # High-end Gen5 NVMe
    
    # Offloading strategies:
    # 1. Synchronous (Block inference)
    # 2. Asynchronous DMA (Overlap with next token generation)
    # 3. Layer-wise Streaming (Stream cache as layers are computed)
    
    token_gen_latency_ms = 20.0 # Time to generate one token
    
    # Results storage
    strategies = ["Synchronous", "Asynchronous DMA", "Layer-wise Streaming"]
    latencies = []
    
    # 1. Synchronous
    sync_latency = (cache_size_gb / pcie_bandwidth_gbps) * 1000 # ms
    latencies.append(sync_latency)
    
    # 2. Async DMA (Assuming we can overlap 90% with compute)
    async_latency = sync_latency * 0.1
    latencies.append(async_latency)
    
    # 3. Layer-wise (Assuming 32 layers, streaming each chunk)
    layer_chunk = cache_size_gb / 32
    layer_latency = (layer_chunk / pcie_bandwidth_gbps) * 1000
    # Ideally, layer_latency < time_to_compute_layer_tokens
    # We'll simulate the effective "stall" time
    layer_stall = max(0, layer_latency - (token_gen_latency_ms / 32))
    latencies.append(layer_stall * 32)

    # Plotting
    plt.figure(figsize=(10, 6))
    colors = ['#ff9999','#66b3ff','#99ff99']
    plt.bar(strategies, latencies, color=colors)
    plt.ylabel('Effective Latency (ms)')
    plt.title('Temporal KV-Cache Offloading Latency (8GB Cache)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    output_dir = "ml-explorations/2026-02-13_temporal-kv-cache-offloading"
    plt.savefig(os.path.join(output_dir, 'offloading_benchmark.png'))
    
    with open(os.path.join(output_dir, "simulation_results.txt"), "w") as f:
        f.write(f"Cache Size: {cache_size_gb} GB\n")
        for s, l in zip(strategies, latencies):
            f.write(f"{s}: {l:.2f} ms\n")

if __name__ == "__main__":
    simulate_offloading()
