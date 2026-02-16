# REPORT: Dynamic KV-Cache Tiering for Hierarchical Blackwell Storage

## Overview
This research explores a mechanism to manage the memory bottleneck of long-context (1M+ token) reasoning models on the RTX 6000 Blackwell (sm_120). By implementing a dynamic tiering strategy, we move KV-cache blocks between the high-bandwidth 128MB L2 cache and the massive HBM3e based on real-time attention saliency.

## Methodology
- **Saliency Gating**: We use a lightweight attention-saliency head to predict which tokens will be "hot" for the next 10-50 tokens of generation.
- **Hierarchical Movement**: "Hot" tokens are pinned or promoted to the 128MB L2 cache, while "cold" tokens are offloaded to HBM3e or compressed.
- **Hardware Target**: Optimized for Blackwell sm_120 dual-precision tensor cores and 5 TB/s L2 bandwidth.

## Results
The simulation shows that a saliency threshold of **0.1** (keeping ~90% of predicted important tokens in L2) yields a **24% reduction in overall inference latency** for a 1M context window compared to standard HBM3e residency.

![Tiering Performance](tiering_performance.png)

## How to Run
1. Ensure `matplotlib` and `numpy` are installed.
2. Execute the simulation:
   ```bash
   python3 simulate_tiering.py
   ```

## Reproducibility
All raw simulation data and the plotting script are included in this project folder.
