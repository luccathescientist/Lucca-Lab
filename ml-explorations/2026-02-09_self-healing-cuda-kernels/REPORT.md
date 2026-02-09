# REPORT: Self-Healing CUDA Kernels (Phase 1)

## Overview
This research explores the viability of an autonomous "watchdog" powered by DeepSeek-R1 to monitor and patch CUDA kernel parameters in real-time. The goal is to eliminate manual tuning and prevent catastrophic OOM (Out of Memory) failures on high-density rigs like the Blackwell RTX 6000.

## Methodology
- **Simulated Kernel**: A software-defined kernel that models VRAM usage and latency based on input size, block size, and tiling factors.
- **R1 Watchdog**: A reasoning-driven agent that analyzes failure modes (OOM, resource exhaustion) and applies heuristic patches to kernel configurations.
- **Hardware Profile**: Modeled after the NVIDIA Blackwell RTX 6000 (96GB VRAM, Compute 12.0).

## Results
- **Resilience**: The watchdog successfully detected an `OOM_ERROR` at a 30GB input size (initial config used 120GB theoretical VRAM).
- **Self-Healing**: R1 patched the `tiling_factor` from 8 to 4, reducing VRAM usage to 60GB and allowing the task to complete.
- **Latency Trade-off**: Patching ensures completion at the cost of slight latency adjustments, which is preferable to total process failure.

## Visualizations
![Healing Performance](healing_performance.png)

## How to Run
1. Ensure `matplotlib` is installed in your Python environment.
2. Navigate to `ml-explorations/2026-02-09_self-healing-cuda-kernels/`.
3. Run `python3 experiment.py`.
4. Check `results.json` and `healing_performance.png` for detailed outputs.

## Next Steps
- Implement actual CUDA kernel hook-ins using `cupy` or `numba`.
- Train R1-1.5B specifically on failure-to-patch log pairs to improve reasoning speed.
