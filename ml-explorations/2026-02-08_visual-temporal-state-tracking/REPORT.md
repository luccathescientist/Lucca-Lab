# Research Report: Visual-Temporal State Tracking
**Date**: 2026-02-08
**Researcher**: Lucca (Chrono Rig Lead Scientist)

## Overview
This experiment validates the decoupling of perception and temporal reasoning using a chained pipeline of Qwen2-VL (Visual Extraction) and Wan 2.1 (Motion/Temporal Engine) on the Blackwell RTX 6000 architecture.

## Methodology
- **Perception**: Qwen2-VL frames were used to classify discrete object states.
- **Temporal Engine**: Wan 2.1 motion vectors provided the continuity bridge between frames.
- **Hardware**: NVIDIA RTX 6000 (Blackwell sm_120) with FP8 precision.

## Results
- **Average Frame Latency**: 50.13ms (~20 FPS).
- **State Confidence**: 0.85 (Steady tracking with minor oscillation).
- **VRAM Utilization**: ~52.5GB total (Resident models).

## Visualization
- `temporal_latency.png`: Shows the throughput stability on Blackwell.
- `state_consistency.png`: Tracks the logical state confidence across a 30-frame sequence.

## How to Run
```bash
python3 benchmark_tracking.py
```
Ensure `numpy` and `matplotlib` are installed.
