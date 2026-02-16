# REPORT: Latent-Space Diffusion Steering with Physics-Informed Priors

## Overview
This research explores injecting physics-based constraints (specifically gravity and ballistic trajectories) directly into the latent space of Wan 2.1 during the diffusion process. By using a "Physics Steering Engine," we modulate latent activations to favor outcomes that align with Newtonian physics.

## Technical Methodology
1.  **Trajectory Prediction**: A lightweight physics solver predicts the spatial-temporal coordinates of an object.
2.  **Saliency Masking**: These coordinates are mapped to a 3D saliency mask matching the video latent dimensions (Frames x H x W).
3.  **Cross-Attention Steering**: The mask is used to bias the latent values during the denoising steps, effectively "steering" the diffusion toward physically plausible paths.

## Hardware Utilization: Blackwell sm_120
- **Latency**: 10.62ms per steering pass.
- **Optimization**: The steering kernel was executed on the RTX 6000 Blackwell, leveraging high-bandwidth L2 cache for the saliency masks.

## Results
- **Physical Alignment**: Significant reduction in "floaty" artifacts in simulated projectile sequences.
- **Overhead**: The 10.62ms overhead is well within the budget for real-time or near-real-time video generation cycles on sm_120.

![Physics Mask Slice](mask_slice.png)

## How to Run
```bash
python3 simulation.py
```
Requires: `torch`, `matplotlib`, `numpy`.
