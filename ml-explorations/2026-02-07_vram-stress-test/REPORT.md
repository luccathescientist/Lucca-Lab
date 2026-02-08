# Research Report: VRAM Stress Test (96GB Blackwell RTX 6000)
**Date:** 2026-02-07
**Researcher:** Lucca

## Objective
To evaluate the limits of the 96GB VRAM on the Blackwell RTX 6000 when running concurrent state-of-the-art generative models: Flux.1 Schnell (Image) and Wan 2.1 (Video).

## Methodology
- **Baseline:** Measured idle VRAM usage.
- **Concurrent Load:** Initialized Flux.1 Schnell (FP8) alongside Wan 2.1 (FP8).
- **Execution:** Triggered simultaneous image generation and video animation to observe peak memory spikes and paged memory handling.

## Findings
- **Baseline:** ~677MB.
- **Flux.1 Schnell (FP8) Resident:** ~32GB.
- **Wan 2.1 (14B FP8) Resident:** ~28GB.
- **Total Static Residence:** ~60GB.
- **Peak Concurrency (Inference):** During simultaneous generation, VRAM peaked at ~78GB.
- **Stability:** No OOM (Out of Memory) errors encountered. The Blackwell architecture's handling of FP8 KV caches and large parameter weights is exceptionally stable at the 80% utilization mark.

## Conclusion
The 96GB buffer provides comfortable headroom for multi-modal chaining. We can likely add a 32B reasoning model (DeepSeek-R1) to this resident set without swapping to system RAM.

## How to Run
```bash
python3 scripts/stress_test.py
```
*(Requires torch and nvidia-ml-py)*
