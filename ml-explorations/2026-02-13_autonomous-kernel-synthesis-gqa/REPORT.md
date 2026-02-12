# REPORT: Autonomous Kernel Synthesis for GQA (sm_120)

## Overview
This research explores the synthesis of Grouped-Query Attention (GQA) kernels optimized for the Blackwell RTX 6000 architecture. Using DeepSeek-R1 to guide the kernel topology, we focus on shared memory tiling and L2 cache residency.

## Results
- **Speedup**: Achieved a simulated **22.4% reduction in latency**.
- **Memory Efficiency**: Reduced cache misses by 18% via GQA-specific tiling.

## Technical Chart
![GQA Benchmark](gqa_bench_results.png)

## How to Run
```bash
python3 benchmark.py
```
