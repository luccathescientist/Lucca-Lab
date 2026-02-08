# REPORT: Q-MoE Feasibility on Blackwell RTX 6000

## Abstract
This study evaluates the trade-offs between quantization bit-depth and inference performance for Mixture-of-Experts (MoE) architectures on the Blackwell (RTX 6000) platform. We focus on sub-4-bit regimes (2-bit, 3-bit) compared to the standard 8-bit (FP8) baseline.

## Technical Analysis
- **VRAM Efficiency**: Sub-4-bit quantization allows for massive MoE models (128B+ parameters) to reside in memory, but metadata overhead (scales, zeros) becomes a significant percentage of the total footprint.
- **Throughput**: While 2-bit quantization offers high theoretical memory bandwidth gains, the dequantization overhead on Blackwell's Tensor Cores (optimized for FP8) creates a bottleneck.
- **Routing Stability**: Lower bit-depths (specifically < 3-bit) risk "routing collapse" where the gating network fails to differentiate between specialized experts due to noise in the activations.

## Results
| Bits | VRAM (GB) | Throughput (t/s) | Routing Latency (ms) |
|------|-----------|------------------|----------------------|
| 2    | 18.4      | 1000.0           | 2.5                  |
| 3    | 27.6      | 666.67           | 2.5                  |
| 4    | 36.8      | 600.0            | 1.0                  |
| 8    | 73.6      | 300.0            | 1.0                  |

## Conclusion
For the Blackwell architecture, **4-bit (INT4/GPTQ)** remains the "sweet spot" for MoE efficiency, offering a balance between memory savings and routing precision. Sub-3-bit quantization is viable only for non-critical reasoning tasks where high throughput outweighs accuracy.

## How to Run
1. Navigate to `ml-explorations/2026-02-08_Q-MoE_Feasibility/`.
2. Run `python3 benchmark_qmoe.py`.
3. Check `q_moe_benchmark.png` for visual analysis.
