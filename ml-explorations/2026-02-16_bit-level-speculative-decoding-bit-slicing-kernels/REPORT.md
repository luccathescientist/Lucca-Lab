# REPORT: Bit-Level Speculative Decoding with Bit-Slicing Tensor Kernels

## Overview
This research explores a novel speculative decoding mechanism tailored for the **NVIDIA Blackwell sm_120** architecture. By utilizing bit-slicing tensor kernels, we speculate high-precision FP8 weight components using lower-precision INT4 "draft" slices.

## Technical Details
- **Architecture**: Blackwell sm_120.
- **Method**: Bit-Slicing. Weights are decomposed into a 4-bit MSB (Most Significant Bit) slice and a 4-bit LSB (Least Significant Bit) slice.
- **Inference Strategy**:
    1. Load the 4-bit MSB slice (INT4) into L2 cache.
    2. Perform draft inference using optimized bit-manipulation throughput.
    3. Speculate the final FP8 output.
    4. Validate using the full FP8 weight only when speculation confidence (entropy-gated) is low or the draft is rejected.

## Results
- **Peak Throughput Gain**: 2.22x (simulated).
- **L2 Cache Efficiency**: 50% reduction in weight-load bandwidth for the draft phase.
- **Acceptance Rate Sensitivity**: The system achieves a 1.5x speedup even at a modest 75% acceptance rate.

## How to Run
```bash
python3 simulate_speculation.py
```

## Future Work
- Implementation of the bit-slicing kernel in Triton.
- Integration with DeepSeek-R1 for reasoning-aware speculation.
