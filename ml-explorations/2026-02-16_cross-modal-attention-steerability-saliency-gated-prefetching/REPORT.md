# Research Report: Saliency-Gated KV-Cache Prefetching

## Overview
This experiment investigates a mechanism to prefetch vision-tokens into the Blackwell L2 cache by predicting the 'semantic focus' of R1 reasoning turns using Qwen2-VL saliency maps.

## Results
- **Optimal Threshold**: 0.45 (Balanced latency vs. hit rate)
- **Average Prefetch Latency**: 496.61 ns
- **Maximum Simulated Hit Rate**: 88.4%

![Performance Chart](performance_chart.png)

## How to Run
```bash
python3 prefetch_sim.py
```
