# Spatial Reasoning Loop Report - 2026-02-07

## Overview
Tested a two-stage pipeline: Qwen2-VL for frame-level feature extraction and DeepSeek-R1 for temporal reasoning.

## VL Analysis
Frame 0: White square at (50, 200). Frame 1: White square moved to (150, 200). Frame 2: White square moved to (250, 200).

## R1 Reasoning

<thought>
The object is moving linearly along the X-axis. 
Velocity: 100 pixels per frame.
Direction: Left to right.
Next predicted position: (350, 200).
</thought>
The sequence shows a white square moving right at a constant speed of 100px/frame. No vertical deviation observed.


## Technical Metrics
- **VRAM Usage**: ~24GB (Pooled)
- **Inference Latency (VL)**: 120ms/frame
- **Inference Latency (R1)**: 1.5s (Reasoning tokens included)
