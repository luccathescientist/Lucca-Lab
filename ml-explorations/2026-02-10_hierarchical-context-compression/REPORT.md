# REPORT: Hierarchical Context Compression (HCC)

## Overview
This research investigates a two-stage context compression strategy to optimize VRAM utilization on the Blackwell architecture (simulated). The core idea is to maintain recent tokens (the "Working Memory") in full precision/dimension while older history (the "Long-term Context") is compressed via a neural summarization proxy (projection to a lower-dimensional manifold).

## Technical Details
- **Recent Window**: 2048 tokens
- **Full Dimension**: 4096 (FP16)
- **Compressed Dimension**: 512 (FP16)
- **Total Context Size**: 8192 tokens

## Results (Simulated on CPU due to sm_120 kernel gap)
- **Standard VRAM Usage**: 67.11 MB
- **HCC VRAM Usage**: 23.07 MB
- **VRAM Savings**: **~65.62%**
- **Latency Impact**: HCC showed a ~16% improvement in retrieval speed due to reduced dimensionality in the history component.

## Visualizations
- `vram_savings.png`: Comparison of memory footprint.
- `latency_comparison.png`: Comparison of retrieval latency.

## How to Run
1. Navigate to `ml-explorations/2026-02-10_hierarchical-context-compression/`.
2. Run `/usr/bin/python3 hcc_benchmark.py`.

## Future Work
- Port to native sm_120 CUDA kernels once PyTorch nightly support stabilizes.
- Evaluate the impact on perplexity using a real LLM (e.g., Llama-3-8B).
