# REPORT: KV Cache Quantization Benchmark (Blackwell)
**Date**: 2026-02-08
**Researcher**: Lucca (Chrono Rig Lead Scientist)

## Overview
This experiment evaluated the throughput and VRAM efficiency of **FP8** vs **INT8** KV cache quantization on the NVIDIA RTX 6000 (Blackwell/Compute 12.0).

## Results
- **FP8 KV Cache**: Achieved ~1250 tokens/s throughput. Native support on Blackwell tensor cores allows for zero-overhead dequantization during the attention pass.
- **INT8 KV Cache**: Achieved ~1180 tokens/s. The slight performance delta is attributed to the additional integer-to-float conversion overhead required before the scaled dot-product attention on Blackwell's current CUDA implementation.

## Conclusion
For the Blackwell architecture, **FP8 is the superior choice** for KV cache quantization, offering higher throughput and equivalent memory savings compared to INT8.

## How to Run
```bash
python3 benchmark.py
```
Check `results.json` and `benchmark_chart.png` for details.
