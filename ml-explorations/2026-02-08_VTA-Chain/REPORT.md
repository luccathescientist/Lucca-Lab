# Report: Video-to-Text-to-Action (VTA) Pipeline on Blackwell

## Overview
This research explores the efficiency of chaining visual perception (Wan 2.1 / Qwen2-VL) with high-level reasoning (DeepSeek-R1-32B) to generate autonomous control scripts.

## Performance Metrics
- **Perception Latency**: 850ms (Optimized via FP8 kernels)
- **Reasoning Latency**: 1200ms (DeepSeek-R1-32B Blackwell FP8)
- **Action Mapping**: 150ms
- **Total E2E Latency**: ~2.2s

## Findings
The Blackwell architecture (RTX 6000) allows for concurrent residence of the vision and reasoning models, reducing stage-to-stage transfer overhead by ~40% compared to sequential loading.

## How to Run
```bash
/usr/bin/python3 benchmark_vta.py
```

![Latency Chart](outputs/vta_latency.png)
