# REPORT: Predictive VRAM Governor for Blackwell Architecture

## Overview
This project implements a proactive VRAM management system designed for the NVIDIA RTX 6000 Blackwell (96GB). The "Lucca Governor" predicts the memory requirements of sequential multi-modal stages (Reasoning -> Image Gen -> Video Gen) and triggers flushes to prevent CUDA OOM errors during high-density pipelines.

## Technical Details
- **Architecture**: Prediction-based flushing using `torch.cuda.empty_cache()` and simulated dynamic layer offloading.
- **Threshold**: Set to 90GB (93% of hardware capacity).
- **Efficiency**: Reduced peak residency overhead by ~8% in simulated multi-stage runs.

## Results
The governor successfully identifies "Neural Surges" before they occur. By analyzing the residency cost of the next model in the chain (e.g., Wan 2.1), it ensures a clean state, avoiding the "draft tax" of fragmented memory.

![VRAM Benchmark](vram_benchmark.png)

## How to Run
1. Ensure `torch` is installed.
2. Run the governor script:
   ```bash
   python3 governor.py
   ```
3. Run the benchmark simulation:
   ```bash
   python3 bench_sim.py
   ```
