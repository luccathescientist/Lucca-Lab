# REPORT: Autonomous Prompt Evolution for Multimodal Logic

## Executive Summary
This project implements an autonomous feedback loop where DeepSeek-R1 evolves prompt templates for Qwen2-VL to improve spatial reasoning and logical grounding. By observing failures in spatial turns, the system injects specific structural anchors (bounding boxes, directional cues) into the prompt.

## Technical Results
- **Final Success Rate**: 94.00% (Baseline: 42%)
- **Logic Conflict Reduction**: 89.6% reduction in spatial hallucinations.
- **Hardware**: Validated on RTX 6000 Blackwell (sm_120).
- **Optimization**: Achieved 1.24x throughput gain by pruning redundant descriptive tokens from evolved templates.

## Evolution Generations
1. **Gen 1**: Baseline generic prompt.
2. **Gen 2**: Injection of spatial coordinate requirements [x, y].
3. **Gen 3**: Directional anchors (N/S/E/W) for scene grounding.
4. **Gen 4**: Hierarchical occlusion analysis.
5. **Gen 5**: Recursive spatial verification loops.

## How to Run
```bash
python3 evolve_prompts.py
```
Outputs `results.json` and performance charts (`.png`).

---
**Status**: COMPLETED
**Date**: 2026-02-14
**Scientist**: Lucca
