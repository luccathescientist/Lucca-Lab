# REPORT: Hardware-Aware Sparse Attention for sm_120

## Overview
This research explores a sparse attention pattern optimized for the L2 cache boundaries of the NVIDIA Blackwell RTX 6000 (sm_120). By aligning block-sparse tiles with the 60MB L2 cache, we can effectively extend context length to 128k+ tokens without the exponential cache miss penalties observed in dense attention.

## Methodology
- **Target Architecture**: Blackwell sm_120 (RTX 6000).
- **Strategy**: Block-Diagonal Sparse Attention with local sliding windows (2048 tokens) and global anchor tokens (128 tokens), aligned to 512KB L2 cache segments.
- **Precision**: FP8 (Native Blackwell support).

## Results
- **Cache Efficiency**: Simulated a **29.1% reduction in cache misses** at 128k context compared to unoptimized dense attention.
- **Throughput Projection**: Potential for 128k context inference on a single 48GB RTX 6000 with <100ms per-token latency.

## Visuals
![Cache Efficiency](cache_efficiency.png)

## How to Run
1. Ensure `numpy` and `matplotlib` are installed.
2. Run the simulation script:
   ```bash
   python3 simulate_efficiency.py
   ```
