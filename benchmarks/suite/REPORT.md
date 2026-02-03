# Blackwell Benchmark Suite: Initial Run

## Objective
The objective of this suite is to automate performance measurement of trending AI models on the **NVIDIA RTX PRO 6000 Blackwell** (96GB VRAM) laboratory workstation.

## Results: 2026-02-03

### Causal LLM Benchmarks (FP16)
| Model | VRAM Usage (GB) | Tokens/Sec | Load Time (s) |
| :--- | :---: | :---: | :---: |
| DeepSeek-R1-Distill-Qwen-1.5B | 3.31 | **110.06** | 3.48 |
| DeepSeek-R1-Distill-Qwen-32B | 61.03 | **20.71** | 32.07 |

### Observations
- **Blackwell sm_120**: The 96GB VRAM capacity allows for large models like the 32B distill without needing aggressive quantization.
- **Inference Efficiency**: The 1.5B model achieves instantaneous response times, making it ideal for real-time agentic logic.

## Environment
- **Torch**: 2.8.0+cu128
- **Hardware**: NVIDIA RTX PRO 6000 Blackwell
- **OS**: Ubuntu 22.04

---
*Report generated automatically by the Blackwell Benchmark Suite v1.0*
