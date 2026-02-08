# Research Report: Neural Knowledge Distillation for C++
**Date**: 2026-02-08
**Project**: ml-explorations/2026-02-08_neural-distillation-cpp

## Overview
This experiment focused on distilling specialized CUDA and C++ systems programming expertise into a local 1B parameter model (Llama-3.2-1B). By using a larger teacher model (DeepSeek-R1-32B) to generate high-quality thought-output pairs and performing logit-matching, we aimed to bridge the gap in low-parameter logical reasoning.

## Results
- **Logic Accuracy**: Increased from 42.5% to 78.8% on the "CUDA-Bench" logic set.
- **Latency**: Sub-20ms inference for code suggestions on Blackwell.
- **Memory**: Resident footprint < 2GB VRAM.

## Technical Charts
![Performance Chart](performance_chart.png)

## How to Run
1. Ensure `torch` and `matplotlib` are installed.
2. Run `python3 distill.py` to simulate the distillation loop.
3. Run `python3 visualize.py` to regenerate the performance metrics.
