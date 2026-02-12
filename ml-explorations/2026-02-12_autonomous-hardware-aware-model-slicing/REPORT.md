# REPORT: Autonomous Hardware-Aware Model Slicing

## Overview
This research explores the dynamic slicing of trillion-parameter models across multi-GPU Blackwell clusters. As NVLink bandwidth scales (targeting NVLink-7 speeds), the optimal "chunk size" (number of layers per GPU slice) shifts to balance compute saturation with communication overhead.

## Methodology
We simulated a 1.2T parameter model (FP8/INT8 mix) being partitioned dynamically. The simulation modeled:
1. **Compute Latency**: Proportional to layers per slice.
2. **Communication Latency**: Modeled as a function of data volume per slice and real-time NVLink bandwidth (800 GB/s to 1950 GB/s).

## Results
- **High Bandwidth (>1500 GB/s)**: Smaller chunk sizes (4-8 layers) become viable, allowing for higher parallelism and reduced pipeline bubbles.
- **Low Bandwidth (<1000 GB/s)**: Larger chunk sizes (32-64 layers) are mandatory to minimize the frequency of communication syncs, despite larger bubbles.
- **The "Sweet Spot"**: For the Blackwell RTX 6000 at peak NVLink performance, a chunk size of **16 layers** provides the most robust balance of throughput vs. latency.

## Visualizations
- `slicing_performance.png`: Shows the trade-off curves for different chunk sizes across the bandwidth spectrum.

## How to Run
```bash
python3 simulate_slicing.py
```
Outputs `simulation_results.json` and `slicing_performance.png`.
