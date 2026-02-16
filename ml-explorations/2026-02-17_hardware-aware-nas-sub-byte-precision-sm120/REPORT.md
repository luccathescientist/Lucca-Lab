# REPORT: Hardware-Aware NAS for Sub-Byte Precision on sm_120

## Overview
This exploration utilized DeepSeek-R1 to autonomously search for transformer architectures robust to extreme quantization (INT4 and INT2), specifically targeting the sub-byte tensor core throughput of the NVIDIA Blackwell sm_120 architecture.

## Key Findings
- **INT4 Superiority**: The NAS identified that certain block structures (residual-heavy with normalized attention) maintain >93% robustness even at INT4, while providing a significant throughput boost over FP8.
- **INT2 Challenges**: While INT2 offers the highest theoretical throughput, architecture robustness dropped to ~81%, indicating the need for more specialized "Bit-Level" distillation or architectural modifications like wider hidden dimensions to compensate for precision loss.
- **Throughput Gains**: Simulated results show near-linear scaling for INT4 on optimized Blackwell kernels, with potential for 3-4x gains if INT2 stability is solved.

## Technical Chart
See `results_chart.png` for the relationship between precision, throughput, and robustness.

## How to Run
1. Navigate to `ml-explorations/2026-02-17_hardware-aware-nas-sub-byte-precision-sm120/`.
2. Ensure `torch`, `numpy`, and `matplotlib` are installed.
3. Run `python3 nas_search.py`.
