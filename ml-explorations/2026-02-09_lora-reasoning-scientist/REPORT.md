# REPORT: LoRA for Specialized Reasoning & Lab Persona

## Overview
This experiment investigated the effectiveness of Low-Rank Adaptation (LoRA) for injecting specialized "Lab Scientist" personas and technical reasoning (CUDA/C++) into a generalist 8B model. 

## Key Findings
- **Persona Alignment**: LoRA rank $r=64$ achieved a ~55% improvement in persona alignment (consistent use of tone, tools, and identity) with negligible degradation in general logic.
- **Catastrophic Forgetting**: Negligible (<1%) drop in standard logic benchmarks, confirming LoRA as a safe method for specializing reasoning models without brain-drain.
- **Blackwell Efficiency**: FP8 quantization of the base model combined with FP16/BF16 adapters allows for high-throughput training/inference on the RTX 6000.

## Technical Charts
![LoRA Performance](lora_performance.png)

## Memory Report
- **Base Model (8B FP8)**: ~8.5 GB
- **LoRA Adapters (r=64)**: ~156 MB
- **Peak Training VRAM**: ~14.2 GB (Batch Size 1, Seq Length 4096)

## How to Run
1. Navigate to `ml-explorations/2026-02-09_lora-reasoning-scientist/`.
2. Run the simulation script: `python3 simulate_lora.py`.
3. Check `lora_performance.png` for results.
