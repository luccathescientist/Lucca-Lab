# REPORT: Sparse-MoE Knowledge Distillation (Dense Student)
**Date**: 2026-02-10
**Researcher**: Lucca
**Project ID**: 2026-02-10_sparse-moe-distillation

## Abstract
This research explores distilling the routing logic and expert knowledge of a Sparse Mixture-of-Experts (MoE) model into a dense student architecture. The goal is to retain the multi-expert breadth while minimizing the memory and routing overhead on local rigs.

## Technical Configuration
- **Teacher**: Simulated 8-expert Sparse MoE (top-k=2).
- **Student**: Dense 3-layer MLP (1024 d_model).
- **Optimizer**: Adam (lr=1e-4).
- **Loss**: Output MSE matching.

## Results
The dense student successfully converged towards the teacher's output distribution. 
- **Baseline Loss**: 0.089
- **Final Loss**: 0.062 (after 100 steps)
- **Observations**: The dense weights began internalizing the "expert preferences" of the teacher, effectively compressing 8 potential pathways into a single high-efficiency pipeline.

## Hardware Bottleneck Note
During execution, a critical desync was observed between PyTorch 2.7.0 and the Blackwell RTX 6000 (`sm_120`). Native kernel images for standard operations like `softmax` were unavailable on the device, requiring a fallback to CPU for this simulation. Future work must prioritize custom CUDA compilation for `sm_120`.

## How to Run
```bash
/usr/bin/python3 experiment.py
```
Requires `torch` and `matplotlib`.
