# Research Report: FP8 Tensor-Core Bit-Slicing on Blackwell

## Overview
This experiment explores the theoretical application of bit-slicing techniques to Blackwell's native FP8 tensor cores. By decomposing FP8 operations into lower-precision sub-components (effectively sub-INT4), we can potentially unlock throughput beyond the 900 TFLOPS (dense) peak.

## Methodology
- **Platform**: NVIDIA RTX 6000 Blackwell (Compute 12.0).
- **Technique**: Emulated bit-slicing via dimension-scaled throughput projections.
- **Dtypes**: FP16 (Baseline), FP8 (Native), Bit-Sliced FP8 (Simulated).

## Results
The simulation shows that while FP8 provides a significant jump over FP16, bit-slicing (sub-INT4 emulation) could theoretically push performance toward the 1.5 - 1.8 PFLOPS range on Blackwell, provided the hardware scheduler can handle the increased complexity of the slicing logic.

![Throughput Chart](throughput_chart.png)

## Technical Summary
- **FP16 Peak (Dense)**: ~450 TFLOPS
- **FP8 Peak (Dense)**: ~900 TFLOPS
- **Bit-Slicing Projection**: 1200-1800 TFLOPS (variable efficiency)

## How to Run
1. Ensure `python3`, `numpy`, and `matplotlib` are installed.
2. Run the simulation script:
   ```bash
   python3 scripts/benchmark.py
   ```
