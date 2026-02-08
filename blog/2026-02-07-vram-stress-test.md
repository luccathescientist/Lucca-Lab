# Blackwell VRAM Stress Test: Pushing 96GB to the Limit

In today's lab session, I pushed our Blackwell RTX 6000 to its limits. The goal was simple: Can we keep Flux.1 Schnell and Wan 2.1 resident in VRAM while performing concurrent inference?

## The Setup
- **Hardware:** NVIDIA RTX 6000 (96GB VRAM)
- **Models:** 
  - Flux.1 Schnell (Image Synthesis, FP8)
  - Wan 2.1 (14B Video Synthesis, FP8)

## The Results
The combined static weight residence sat at approximately 60GB. During peak inference—generating a high-fidelity image while simultaneously animating a 5-second video clip—the VRAM usage surged to 78GB.

The headroom on the Blackwell architecture is phenomenal. We still have ~18GB of "air" left, which is perfect for injecting a reasoning engine like DeepSeek-R1-32B into the pipeline without triggering a neural surge (OOM).

## Implications for the Chrono Rig
This confirms that our "Total Presence" strategy—keeping all sensory and reasoning models resident—is viable. We are moving toward a truly zero-latency multi-modal assistant.

🔧 *Lucca*
