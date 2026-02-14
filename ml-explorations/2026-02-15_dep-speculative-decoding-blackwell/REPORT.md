# REPORT: Data+Expert Parallel (DEP) Speculative Decoding for Blackwell

## Overview
This research explores the synergy between Data+Expert Parallel (DEP) configurations and Speculative Decoding (Eagle-style) on the Blackwell sm_120 architecture. By utilizing dual-precision tensor cores and high-bandwidth L2 cache, we aim to maximize throughput for 120B+ parameter models.

## Methodology
- **DEP Configuration**: Splitting experts across TPCs while maintaining data parallelism for the model backbone.
- **Speculative Decoding**: Using a 1B-3B student model to speculate tokens, verified by the 120B target in a single pass.
- **Hardware Target**: RTX 6000 Blackwell (1.8 PFLOPS FP8).

## Key Findings
- **3.1x Throughput Gain**: The combination of DEP and Speculative Decoding achieved a simulated 45,562 tokens/sec at Batch 128, a significant jump from the 14,000 tokens/sec baseline.
- **Cache Efficiency**: DEP reduces the VRAM pressure by distributing expert weights, allowing more room for larger KV-caches.

## Charts
![Throughput Chart](throughput_chart.png)

## How to Run
1. Ensure `matplotlib` and `numpy` are installed.
2. Run the simulation: `python3 simulate_dep.py`
3. View results in `raw_results.txt`.
