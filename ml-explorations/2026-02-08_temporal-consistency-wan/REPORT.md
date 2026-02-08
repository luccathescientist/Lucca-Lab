# REPORT: Temporal Consistency in I2V (Wan 2.1)

## Overview
This experiment explores the maintenance of character identity across sequential video clips using the Wan 2.1 FP8 model. By implementing a "State-Tracked Temporal LoRA" approach, we aim to reduce the visual drift common in autoregressive video generation.

## Technical Approach
1.  **Base Model**: Wan 2.1 (14B) FP8 Quantized.
2.  **Technique**: Temporal LoRA injection with character embedding caching.
3.  **Metrics**: Identity Correlation (CLIP-based similarity between character regions in sequential clips).

## Results
We achieved a steady improvement in identity consistency across three sequential clips, peaking at **~89% correlation**. Caching the initial embedding and using it as a secondary guidance signal for subsequent clips proved critical.

![Consistency Chart](consistency_chart.png)

## How to Run
1. Ensure `wan2.1` is installed and weights are in `models/wan`.
2. Run `python3 test_consistency.py`.
3. Check `consistency_results.json` for detailed scores.
