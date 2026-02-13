# REPORT: Temporal KV-Cache Offloading for 1M+ Context Multi-Agent Loops

## Abstract
This research investigates an asynchronous DMA (Direct Memory Access) offloading strategy to manage the KV-caches of multiple active reasoning agents on the NVIDIA Blackwell architecture (sm_120). By leveraging PCIe Gen5 bandwidth and overlapping memory transfers with compute, we demonstrate a path toward massive context windows (1M+ tokens) without stalling inference.

## Methodology
We simulated three offloading strategies for an 8GB KV-cache (representative of ~1M tokens in FP16 for a mid-sized model):
1. **Synchronous**: Blocking the inference engine until the transfer is complete.
2. **Asynchronous DMA**: Overlapping the transfer with active token generation.
3. **Layer-wise Streaming**: Streaming KV-cache chunks per-layer during the forward pass.

## Results
- **Synchronous Latency**: ~137.93 ms (significant bottleneck).
- **Asynchronous DMA Latency**: ~13.79 ms (90% overlap achieved).
- **Layer-wise Streaming**: ~1.3 ms effective stall (near-zero overhead).

The simulation confirms that **Layer-wise Streaming** combined with **Asynchronous DMA** is the optimal path for Blackwell's high-bandwidth PCIe Gen5 interface.

![Offloading Benchmark](offloading_benchmark.png)

## How to Run
1. Ensure `matplotlib` and `numpy` are installed.
2. Run `python3 simulate_offloading.py`.
3. View the generated `offloading_benchmark.png` and `simulation_results.txt`.

## Hardware Specifications
- **GPU**: NVIDIA RTX 6000 Blackwell (sm_120)
- **Bus**: PCIe Gen5 x16
- **System Memory**: 128GB DDR5
