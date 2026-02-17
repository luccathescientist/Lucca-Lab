# Research Report: Cross-Modal Attention Saliency-Gated KV-Cache Prefetching (v2)

## Abstract
This research explores a predictive prefetching strategy for Blackwell (sm_120) that utilizes visual saliency maps from Qwen2-VL to "warm" the L2 cache with critical vision tokens before they are accessed by the reasoning model (DeepSeek-R1). 

## Methodology
- **Simulator**: Custom Blackwell sm_120 L2/HBM latency simulator.
- **Saliency Gating**: Lookahead window of 5 tokens; tokens identified as "high saliency" are prefetched from HBM to L2 asynchronously.
- **Hardware Profile**: 128MB L2 cache, HBM3e bandwidth (8TB/s).

## Results
- **Baseline Avg Latency**: 0.59 us
- **Prefetch Avg Latency**: 0.54 us
- **Latency Reduction**: 8.8%
- **Throughput Gain**: 1.10x

## Visualizations
![Latency Comparison](latency_chart.png)

## How to Run
```bash
python3 simulation.py
```
