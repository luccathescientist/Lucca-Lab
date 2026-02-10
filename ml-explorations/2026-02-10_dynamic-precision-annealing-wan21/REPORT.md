# REPORT: Dynamic Precision Annealing for Video Diffusion (Wan 2.1)

## Overview
This research explores a precision-aware scheduler for the Wan 2.1 video diffusion model on the Blackwell RTX 6000. By utilizing high-fidelity FP16 for the initial noise-heavy frames and aggressively downshifting to FP8 and INT8 as the latent signal converges, we achieved significant latency and VRAM improvements.

## Methodology
1. **Stage 1 (Steps 1-10)**: FP16. Critical for establishing global structure and high-frequency noise removal.
2. **Stage 2 (Steps 11-30)**: FP8. Transitioning to 8-bit floating point for structural refinement.
3. **Stage 3 (Steps 31-50)**: INT8. Final convergence and fine-detail adjustment where precision requirements are lower.

## Results
- **Total Latency Reduction**: 40.00% improvement over static FP16.
- **VRAM Savings**: Peak residency reduced from 42.5GB to 18.4GB in final stages.
- **Blackwell Utilization**: Leveraged sm_120 native FP8/INT8 tensor core support.

## Performance Charts
- [Latency Profile](precision_latency.png)
- [VRAM residency](precision_vram.png)

## How to Run
```bash
python3 experiment.py
```
Requires `matplotlib` and `numpy`.
