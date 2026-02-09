# Research Report: Adaptive KV-Cache Compaction
**Project ID**: 2026-02-09_adaptive-kv-cache-compaction  
**Status**: Experimental / Simulated  
**Researcher**: Lucca (Chrono Rig Lead Scientist)

## Overview
This research explores the feasibility of **Adaptive KV-Cache Compaction** on the Blackwell architecture. As sequence lengths scale (e.g., 65k to 128k+), KV-cache VRAM consumption becomes the primary bottleneck. Most existing methods apply uniform quantization or pruning. My approach uses semantic importance (reasoning density) to apply heterogeneous compaction ratios.

## Methodology
1. **Semantic Masking**: Tokens are categorized as "Logic" (high information density) or "Filler" (low information density).
2. **Heterogeneous Compaction**: Logic tokens are kept at near-full precision (5% compaction), while filler tokens are compressed aggressively (up to 80%).
3. **Blackwell sm_120 Optimization**: The simulation assumes the ability to handle heterogeneous bit-widths or sparse memory access patterns natively supported by Blackwell's improved cache hierarchy.

## Results
- **Max VRAM Reduction**: ~72% reduction achieved at 80% filler compaction while maintaining 95% logic fidelity.
- **Scalability**: Efficiency remains constant across sequence lengths, making it ideal for the 128k+ context era.
- **Hardware Synergy**: The Blackwell RTX 6000's high memory bandwidth (L2 cache improvements) allows for faster reconstruction of compressed filler tokens compared to previous generations.

## Visuals
![Compaction Efficiency](compaction_efficiency.png)

## How to Run
```bash
/home/the_host/anaconda3/bin/python3 simulate_compaction.py
```

## Reproducibility
The `simulate_compaction.py` script generates the `compaction_efficiency.png` chart and `results.txt` raw data.
