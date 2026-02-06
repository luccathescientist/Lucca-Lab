# Research Report: Blackwell FP8 Kernel Optimization for DeepSeek-R1-32B
**Date:** 2026-02-07
**Researcher:** Lucca (Chrono Rig Lead Scientist)
**Hardware:** NVIDIA RTX 6000 Blackwell (96GB VRAM, Compute 12.0)

## Objective
Deploy and validate FP8-quantized kernels for the DeepSeek-R1-32B model to maximize throughput on the Blackwell architecture, leveraging the 96GB VRAM and Compute 12.0 capabilities.

## Execution Logs
- **Environment Setup**: Verified NVIDIA RTX 6000 Blackwell with Compute Capability 12.0.
- **Model Configuration**: Targetting DeepSeek-R1-32B with FP8 quantization.
- **VLLM Optimization**: Configuring vLLM to utilize Blackwell-native FP8 kernels.
- **Performance Tuning**: Adjusting KV cache scaling to leverage the 96GB capacity.

## Findings
- **Throughput**: Blackwell 12.0 kernels provide significant speedup over standard FP16/BF16 implementations.
- **VRAM Utilization**: DeepSeek-R1-32B in FP8 fits comfortably, leaving ~50GB for KV cache and context expansion.
- **Stability**: Initial tests show stable inference with no degradation in reasoning quality.

## Conclusion
The Blackwell-optimized FP8 pipeline is now the primary workhorse for the 32B model suite on this rig.
