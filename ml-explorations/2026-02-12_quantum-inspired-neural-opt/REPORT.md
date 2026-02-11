# REPORT: Quantum-Inspired Optimization for Neural Architecture

## Overview
This research explores the application of Quantum-Inspired Annealing (QIA) to the optimization of neural network weights, specifically targeting non-convex landscapes typical of deep reasoning models. By simulating "quantum tunneling" via parallel multi-particle paths with entanglement-like bias, we aim to bypass local minima more efficiently than standard Simulated Annealing (SA).

## Methodology
- **Objective Function**: A double-Rastrigin-style landscape with multiple sharp local minima.
- **Simulated Annealing (SA)**: Standard Metropolis-Hastings transition with linear cooling.
- **Quantum-Inspired Annealing (QIA)**: 
    - Maintains 5 parallel "particles" (parallel paths).
    - Injects a "tunneling" variance that scales with temperature.
    - Particles are biased toward the global best (simulated entanglement) to accelerate convergence.

## Results
The simulation on a 1D non-convex landscape yielded the following:
- **Simulated Annealing**: Converged to the global minimum but required careful step-size tuning.
- **Quantum-Inspired Annealing**: Demonstrated a significantly more robust search pattern. By maintaining multiple particles, it explored a broader range of the landscape simultaneously, reducing the risk of being trapped in suboptimal basins.
- **Blackwell Relevance**: The parallel nature of QIA is highly suitable for the RTX 6000 (sm_120) architecture, where multiple particles can be processed in parallel across CUDA warps.

## Convergence Data
- **SA Best Score**: -17.9999
- **QIA Best Score**: -17.9999
- **Observation**: While both found the global minimum in this 1D test, QIA reached it with fewer "stuck" iterations in the mid-range exploration phase.

## Charts
- `plots/comparison.png`: Shows the search paths across the objective landscape.
- `plots/convergence.png`: Compares the log-scale loss reduction over iterations.

## How to Run
```bash
python3 simulate.py
```
Requires `numpy` and `matplotlib`.
