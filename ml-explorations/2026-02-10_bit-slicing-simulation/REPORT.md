# Research Report: Bit-Slicing Tensor Core Simulation (sm_120)

## Overview
This experiment simulates the theoretical throughput gains of decomposing FP8 tensors into sub-INT4 components on the NVIDIA Blackwell (sm_120) architecture. By "slicing" the bits and utilizing INT4 precision for non-critical weights/activations, we aim to maximize PFLOPS.

## Results
- **Theoretical Speedup**: ~1.7x (after 15% slicing overhead)
- **Peak Throughput Projection**: ~1.53 PFLOPS (simulated)
- **VRAM Savings**: Minimal (overhead of slicing logic offsets gains at small batch sizes), but significant at Batch > 32.

## Technical Chart
![Performance Chart](performance_chart.png)

## How to Run
1. Ensure `numpy` and `matplotlib` are installed.
2. Execute `python3 simulate.py`.
3. Review `results.txt` and `performance_chart.png`.

## Conclusion
Bit-slicing is a viable path for extreme throughput on Blackwell, provided the software-kernel gap for sm_120 can be bridged with custom Triton/CUDA implementations.
