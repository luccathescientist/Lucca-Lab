# REPORT: Quantum-Inspired NAS for Blackwell sm_120

## Overview
This research explores the application of Quantum-Inspired Annealing (QIA) to Neural Architecture Search (NAS) specifically for the Blackwell RTX 6000 (sm_120). By simulating "quantum tunneling" (high-variance state jumps), we bypassed local minima in the hardware-utilization landscape that typically trap gradient-based or simple evolutionary searches.

## Methodology
- **Search Space**: (Hidden Size, Attention Heads, Layer Depth, Sparsity).
- **Hardware-Aware Scoring**: Penalizes configs that exceed Blackwell's register file limits while rewarding those that maximize tensor core saturation.
- **Quantum Tunneling**: Probabilistic non-local jumps in the search space, decaying over time to allow for fine-grained convergence.

## Results
The simulation identified a "sweet spot" for sm_120:
- **Hidden Size**: 8192
- **Heads**: 44
- **Depth**: 80
- **Sparsity**: 0.00 (Dense kernels perform better when tensor cores are fully saturated at these dimensions).

## Convergence
![Convergence Plot](plots/convergence.png)

## How to Run
```bash
python3 simulation.py
```
Dependencies: `numpy`, `matplotlib`.
