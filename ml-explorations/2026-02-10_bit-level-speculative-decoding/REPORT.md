# REPORT: Bit-Level Speculative Decoding on Blackwell

## Overview
This research explores the efficiency of using **Bit-Level Speculative Decoding** (INT4 drafts) to accelerate high-precision (FP8) inference on the NVIDIA RTX 6000 Blackwell architecture.

## Hypothesis
By using a sub-FP8 (INT4) draft model, we can reduce the memory bandwidth bottleneck during the auto-regressive decoding phase while leveraging Blackwell's native tensor core support for fast target-model verification.

## Results
- **Baseline FP8 Latency**: 1.0000 units/token
- **INT4 Speculative Latency**: 0.6752 units/token
- **Observed Speedup**: **1.48x**

The simulation demonstrates that even with a moderate acceptance rate (~72%), the reduced compute/memory cost of the INT4 draft model significantly outweighs the "draft tax," leading to a near 50% throughput increase.

## Technical Chart
![Speedup Chart](speedup_chart.png)

## How to Run
1. Ensure `matplotlib` is installed.
2. Run `python3 simulate.py` to generate the raw data.
3. Run `python3 plot_results.py` to generate the speedup chart.

## Conclusion
Bit-level speculation is a high-impact optimization for local rigs where VRAM bandwidth is the primary constraint. Future work will focus on native `sm_120` kernel implementations for these bit-sliced operations.
