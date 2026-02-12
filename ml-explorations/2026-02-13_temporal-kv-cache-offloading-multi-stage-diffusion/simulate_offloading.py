import time
import numpy as np
import matplotlib.pyplot as plt

def simulate_offloading():
    # Model sizes (simulated in GB)
    flux_vram = 24  # Flux.1 image model
    wan_vram = 32   # Wan 2.1 video model
    kv_cache_size = 8 # KV cache to be offloaded
    
    total_vram = 80 # RTX 6000 Blackwell
    
    # Bandwidths (simulated in GB/s)
    pcie_gen5_bw = 63 # GB/s (PCIe 5.0 x16)
    nvme_bw = 14 # GB/s (Gen5 NVMe)
    gpu_copy_bw = 900 # GB/s (Internal HBM3e)
    
    results = {
        "PCIE_Gen5": [],
        "NVMe_Direct": [],
        "Standard_RAM": []
    }
    
    cache_sizes = [1, 2, 4, 8, 16, 32] # GB
    
    for size in cache_sizes:
        # Time = Size / Bandwidth
        results["PCIE_Gen5"].append(size / pcie_gen5_bw * 1000) # ms
        results["NVMe_Direct"].append(size / nvme_bw * 1000)   # ms
        results["Standard_RAM"].append(size / (pcie_gen5_bw * 0.5) * 1000) # ms (assumed overhead)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(cache_sizes, results["PCIE_Gen5"], label='PCIe Gen5 (63 GB/s)', marker='o')
    plt.plot(cache_sizes, results["NVMe_Direct"], label='NVMe Direct (14 GB/s)', marker='s')
    plt.plot(cache_sizes, results["Standard_RAM"], label='Standard RAM Offload (~30 GB/s)', marker='^')
    
    plt.title('KV-Cache Offloading Latency: PCIe vs NVMe')
    plt.xlabel('KV-Cache Size (GB)')
    plt.ylabel('Latency (ms)')
    plt.legend()
    plt.grid(True)
    plt.savefig('ml-explorations/2026-02-13_temporal-kv-cache-offloading-multi-stage-diffusion/latency_chart.png')
    
    # Print technical stats for REPORT.md
    print(f"Simulation Results for {kv_cache_size}GB KV-Cache:")
    print(f"PCIe Gen5 Latency: {kv_cache_size / pcie_gen5_bw * 1000:.2f} ms")
    print(f"NVMe Direct Latency: {kv_cache_size / nvme_bw * 1000:.2f} ms")
    print(f"Theoretical Blackwell Throughput: {gpu_copy_bw} GB/s")

if __name__ == "__main__":
    simulate_offloading()
