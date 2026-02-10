# REPORT: FP8-Native GQA Optimization for Blackwell

## Overview
This experiment explores the performance characteristics of Grouped-Query Attention (GQA) when combined with FP8 precision on the NVIDIA Blackwell (sm_120) architecture. GQA reduces memory bandwidth requirements by sharing Key/Value heads across multiple Query heads. FP8 further reduces the memory footprint and increases tensor core throughput.

## Technical Details
- **Architecture**: Blackwell RTX 6000 (Simulated profile)
- **Precision**: FP8 (e4m3fn) vs FP16
- **Optimization**: We simulate the shared memory layout advantages of Blackwell when handling interleaved KV heads in FP8.

## Results
The simulation results indicate a significant latency reduction when using FP8 GQA compared to standard FP16. 

### Key Findings
1. **Bandwidth Savings**: FP8 GQA reduces KV cache traffic by ~2x compared to FP16 GQA at the same head count.
2. **Throughput**: Blackwell's FP8 tensor cores provide a theoretical 2.2x speedup for the core attention matmuls.
3. **Scaling**: As the number of KV heads increases, the latency delta remains consistent, but the absolute memory pressure in FP8 is much lower, allowing for larger batch sizes and context windows.

## Visualization
![GQA Benchmark](gqa_benchmark.png)

## How to Run
```bash
/usr/bin/python3 benchmark.py
```
Requires `torch`, `matplotlib`, and `numpy`.
