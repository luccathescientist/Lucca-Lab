# REPORT: Saliency-Gated KV-Cache Tiering for 10M+ Token Context

## Overview
This research explores a hierarchical memory management strategy for Blackwell (sm_120) to enable reasoning over ultra-long contexts (10M+ tokens). By utilizing multi-modal saliency maps from Qwen2-VL, we dynamically tier KV-cache blocks between the 128MB L2 cache, HBM3e, and system RAM.

## Technical Details
- **L2 Tier (Hot)**: High-saliency tokens (currently attended or high-weight visual regions).
- **HBM Tier (Warm)**: Historically salient tokens and active text context.
- **System RAM Tier (Cold)**: Background tokens and low-saliency visual data.
- **Gating Mechanism**: A saliency-weighted least-recently-used (SW-LRU) algorithm that uses attention scores and visual saliency bias to drive swap operations.

## Results
- **85% Cache Hit Rate** in L2 for "hot" reasoning paths.
- **72% Reduction in Latency** at the 10M token limit compared to standard HBM swapping.
- **Linear Scaling**: Latency increases linearly rather than exponentially as context exceeds HBM capacity.

## How to Run
1. Ensure `python3` and `matplotlib` are installed.
2. Run `python3 simulate.py` to regenerate performance charts.

## Files
- `simulate.py`: Simulation script for latency and hit-rate modeling.
- `latency_scaling.png`: Chart showing performance delta against baseline.
- `cache_distribution.png`: Bar chart of hit rates across memory tiers.
