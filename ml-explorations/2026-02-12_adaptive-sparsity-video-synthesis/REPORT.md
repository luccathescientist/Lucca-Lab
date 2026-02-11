# Adaptive Sparsity for Real-Time Video Synthesis

## Overview
This research explores a dynamic pruning mechanism for Wan 2.1 on Blackwell (sm_120). By scaling sparsity relative to the movement complexity of the scene, we achieve significant latency and VRAM savings during static or low-motion frames.

## Technical Details
- **Architecture**: Adaptive Sparsity Controller (ASC) integrated into the 3D-Attention blocks of Wan 2.1.
- **Hardware**: Optimized for Blackwell's 2:4 structured sparsity and fine-grained weight pruning.
- **Metric**: Target Sparsity = 1.0 - (Movement_Complexity * 0.9 + 0.1).

## Results (Simulated)
- **Peak Speedup**: ~2.8x during low-motion scenes (90% sparsity).
- **Average VRAM Reduction**: ~55% (from 40GB to ~18GB average).
- **Latency**: Sub-20ms inference potential on RTX 6000 for 720p resolution during high-sparsity phases.

## How to Run
1. Install requirements: `pip install matplotlib numpy`
2. Execute the simulation: `python3 simulation.py`
3. Check `performance_chart.png` for results.

## Reproducibility
The `simulation.py` script contains the full logic used for these projections.
