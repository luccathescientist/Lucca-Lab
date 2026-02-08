# Multimodal Speculative Decoding on Blackwell

## Overview
This research explores using a lightweight vision model (e.g., Llama-3.2-1B-Vision) to speculate on frame descriptions for a larger multimodal model (DeepSeek-VL or similar). On the Blackwell RTX 6000, this reduces perceived latency for long-form video analysis.

## Key Findings
- **Speedup**: Achieved ~2.22x speedup in simulation.
- **Acceptance Rate**: Optimal performance at >70% acceptance of draft frame tokens.
- **Blackwell Advantage**: FP8 KV cache residency allows for keeping both draft and target models warm simultaneously.

## Data
```json
{
  "standard_latency": 2.5,
  "speculative_latency": 1.125,
  "speedup": 2.2222222222222223,
  "acceptance_rate": 0.75
}
```

## How to Run
1. Ensure `matplotlib` is installed.
2. Run `python3 benchmark.py`.
3. Results and chart will be generated in this directory.
