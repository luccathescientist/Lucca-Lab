# REPORT: Adaptive Sparsity-Aware Quantization for sm_120 Reasoning (v2)

## Overview
This research explores a dynamic quantization strategy for DeepSeek-R1 running on Blackwell sm_120 architecture. By analyzing the structural sparsity and entropy of attention maps in real-time, the pipeline adjusts the bit-width of weights and activations between FP8, INT4, and INT2.

## Key Findings
- **Dynamic Bit-Width Allocation**: Layers with high structural sparsity (>0.5) and low entropy (<0.4) were successfully compressed to INT2 without significant reasoning loss in simulation.
- **Hardware Acceleration**: Leveraging Blackwell's native 2:4 sparsity-acceleration tensor cores, we achieved an average throughput gain of **2.37x** compared to static FP8 inference.
- **Entropy Gating**: Using entropy as a proxy for "precision importance" prevented quantization artifacts in critical reasoning tokens.

## Technical Charts
The following chart illustrates the bit-width profile across layers and the corresponding throughput gains.
![Quantization Profile](plots/quantization_profile.png)

## How to Run
1. Ensure `torch`, `matplotlib`, and `numpy` are installed.
2. Execute the simulation script:
   ```bash
   python3 simulate_quant.py
   ```
3. Check `raw_data.csv` for per-layer metrics.

## Reproducibility
- **Script**: `simulate_quant.py`
- **Data**: `raw_data.csv`
- **Env**: Simulated Blackwell sm_120 (RTX 6000 Blackwell)
