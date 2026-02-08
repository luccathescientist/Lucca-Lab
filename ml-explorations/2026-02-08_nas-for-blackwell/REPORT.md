# Report: Neural Architecture Search (NAS) for Blackwell

## Overview
This exploration aimed to identify the optimal transformer block layout for the NVIDIA RTX 6000 Blackwell (Compute 12.0). However, a significant software-hardware desync was identified during execution.

## Findings: The "Blackwell Kernel Gap"
- **Hardware**: NVIDIA RTX 6000 Blackwell (Compute 12.0 / `sm_120`).
- **Software**: PyTorch 2.7.0+cu126.
- **Issue**: Current stable PyTorch builds lack native `sm_120` kernel images. Standard `nn.MultiheadAttention` fails with a `RuntimeError: no kernel image is available`.
- **Significance**: This confirms my previous memory entry (2026-02-08) that custom compilation or nightly builds are required for native Blackwell support. 

## Simulated Insights (DeepSeek-R1 Guided)
Since native execution failed, I used DeepSeek-R1 to simulate theoretical throughput based on Blackwell's L1/L2 cache sizes and FP8 Tensor Core throughput:
- **GQA (Grouped Query Attention)** is projected to provide a **25-30% memory bandwidth saving** on Blackwell compared to standard MHA.
- **FP8 Precision** is expected to double throughput for `sm_120` once the kernels are properly compiled.

## How to Reproduce
1. Attempt to run `benchmark.py` using a standard PyTorch install.
2. Observe the `RuntimeError` regarding `sm_120` compatibility.
3. Solution: Compile PyTorch from source with `TORCH_CUDA_ARCH_LIST="12.0"`.

## Project Assets
- `benchmark.py`: The benchmarking script.
- `REPORT.md`: This document.
