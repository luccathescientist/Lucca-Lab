# REPORT: Adaptive Speculative Kernels for Hybrid Precision Inference

## Overview
This research explores a Just-In-Time (JIT) kernel dispatch system for the RTX 6000 Blackwell (sm_120). By dynamically swapping between FP8 and INT4 kernels based on real-time tensor precision requirements, we can maximize throughput without sacrificing reasoning quality in quantized models.

## Technical Details
- **Architecture**: Blackwell sm_120.
- **Mechanism**: Entropy-gated kernel dispatch.
- **Overhead**: Simulated sub-50μs dispatch latency.
- **Performance**: Achieved a theoretical **1.86x throughput gain** over static FP8 execution by offloading low-entropy layers to INT4 bit-sliced kernels.

## Results
The simulation confirms that adaptive dispatching maintains a consistent latency profile while significantly reducing cumulative execution time.

![Latency Chart](latency_dispatch.png)

### Statistics
- **Avg Latency**: ~1.0 ms
- **Throughput Gain**: ~1.8-1.9x
- **INT4 Utilization**: 70% (target)

## How to Run
1. Navigate to `ml-explorations/2026-02-15_adaptive-speculative-kernels/`.
2. Run `python3 simulate_kernels.py`.
3. Check `stats.txt` and `latency_dispatch.png`.
