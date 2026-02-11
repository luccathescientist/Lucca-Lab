# REPORT: Hardware-Aware NAS for sm_120

## Overview
This research explores the optimal architectural parameters (hidden size, attention heads) for transformer blocks targeting the Blackwell RTX 6000 (sm_120). By simulating hardware utilization based on peak TFLOPS, memory bandwidth, and occupancy constraints, we identify the "sweet spot" for next-gen reasoning models.

## Results
- **Optimal Hidden Size**: 4096
- **Peak Utilization**: 90.0% (estimated)
- **Observations**: Hidden sizes below 2048 fail to saturate Blackwell's massive tensor core arrays, leading to low utilization. Conversely, sizes above 8192 introduce significant register pressure, potentially reducing occupancy and increasing latency beyond the linear scaling of FLOPs.

## How to Run
```bash
python3 nas_search.py
```
Check `nas_results.png` for the performance trade-off visualization.
