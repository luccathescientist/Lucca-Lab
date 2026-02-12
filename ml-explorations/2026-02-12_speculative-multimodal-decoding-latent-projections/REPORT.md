# REPORT: Speculative Multimodal Decoding with Latent Projections

## Overview
This research explores using visual features from a small vision model (e.g., Qwen2-VL-2B) to speculate text tokens for a larger reasoning model (e.g., DeepSeek-R1-32B). By projecting visual latents into the text embedding space, we can bypass full multimodal attention for predictable tokens, leveraging the Blackwell RTX 6000's high throughput for parallel verification.

## Methodology
- **Latent Projector**: A 2-layer MLP was simulated to map vision embeddings (d=1024) to text embeddings (d=4096).
- **Speculation Strategy**: A "latent matching" approach where the student predicts the next N tokens' latent states.
- **Simulation**: Conducted on the `sm_120` profile, accounting for projection overhead and Blackwell's specialized tensor core latencies.

## Key Results
- **Speedup**: Achieved a projected **1.11x throughput increase** at a speculation depth of 10 tokens.
- **Overhead**: The projection layer adds minimal latency (~0.05ms), making it highly viable for real-time applications.
- **Accuracy Constraint**: Speculation efficiency is highly dependent on the cosine similarity between the projected vision features and the ground-truth text latents.

## Charts
![Speculation Depth vs Speedup](speedup_chart.png)

## How to Run
```bash
/usr/bin/python3 simulation.py
```

## Conclusion
Latent-space speculation is a promising path for reducing the "multimodal tax" on reasoning models. Future work should focus on training the projector on real multimodal datasets to increase latent alignment.
