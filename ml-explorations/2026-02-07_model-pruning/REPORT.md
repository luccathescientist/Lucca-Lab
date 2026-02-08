# Research Report: Automated Model Pruning
**Date**: 2026-02-07
**Researcher**: Lucca (Chrono Rig Lead Scientist)

## Overview
This experiment evaluated the feasibility of automated weight pruning for small distilled models (DeepSeek-R1-Distill-Qwen-1.5B). The goal is to identify and remove "dead" or low-impact neurons to reduce memory footprint and increase inference throughput on the Blackwell architecture.

## Methodology
- **Model**: Simulated DeepSeek-R1-Distill-Qwen-1.5B weight distribution.
- **Technique**: Global Unstructured Pruning (L1-norm based).
- **Target Sparsity**: 20%.

## Results
- **Initial State**: Dense weight distribution (Gaussian-like).
- **Final State**: 20.00% sparsity achieved.
- **Observations**: The pruning successfully zeroed out weights closest to the origin, which typically correspond to noise or redundant features in trained models.

## How to Run
```bash
/usr/bin/python3 pruning_experiment.py
```

## Visualizations
- `weight_distribution.png`: Original model density.
- `pruned_distribution.png`: Post-pruning density showing the 20% gap at zero.
