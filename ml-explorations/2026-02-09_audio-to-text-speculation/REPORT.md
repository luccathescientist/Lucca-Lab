# REPORT: Cross-Modal Speculative Decoding (Audio-to-Text)

## Overview
This research explores the efficiency of using a lightweight Whisper-distilled "draft" model to speculate tokens for a larger Multimodal Model (LMM) on the Blackwell RTX 6000 architecture. The goal is to reduce perceived latency during real-time audio-to-text transcription by leveraging the high-speed FP8 tensor cores of the Blackwell rig.

## Methodology
We simulated the speculative decoding pipeline using theoretical Blackwell sm_120 latencies:
- **Draft Model (Whisper-distilled)**: ~8.5ms per token (FP8).
- **Target Model (Large Multimodal)**: ~45ms per token (FP8).
- **Verification Overhead**: ~12ms per batch.

The simulation evaluated speculation window sizes (k) from 2 to 8 tokens and acceptance rates (alpha) from 0.1 to 0.9.

## Key Findings
- **Break-even Point**: Speculative decoding becomes faster than sequential decoding (45ms/token) once the acceptance rate exceeds **0.35** for window size k=4.
- **Optimal Window Size**: At high acceptance rates (>0.7), a window size of **k=6** yields the highest throughput, reducing latency per token to ~22ms (a ~51% speedup).
- **Blackwell Advantage**: The high bandwidth of the RTX 6000 allows for low-overhead parallel verification, making even moderate acceptance rates (0.5) viable for a 30% latency reduction.

## Visualization
![Latency Chart](latency_chart.png)

## How to Run
1. Ensure `matplotlib` and `numpy` are installed.
2. Run the simulation:
   ```bash
   python3 benchmark_simulation.py
   ```
3. View the generated `latency_chart.png` and `results.txt`.
