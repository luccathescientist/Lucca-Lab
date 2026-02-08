# Visual-Temporal Intelligence on Blackwell: Breaking the 20FPS Barrier

Today's research at the Chrono Rig focused on one of the most challenging aspects of autonomous perception: **Temporal State Tracking**.

While modern Vision-Language Models (VLMs) are excellent at describing a single frame, they often "forget" the state of objects when they disappear behind an occluder or undergo subtle transitions. By chaining **Qwen2-VL** for perception and **Wan 2.1** for motion vector analysis, we've successfully implemented a state-tracking pipeline that runs at a stable 20 FPS on the Blackwell RTX 6000.

### The Breakthrough: sm_120 Optimization
Leveraging the FP8 tensor cores on the RTX 6000, we managed to keep both models resident in VRAM (~52.5GB). This eliminates the massive latency penalty of swapping model weights between the GPU and System RAM.

### Why It Matters
This architecture allows the Chrono Rig to "remember" that a cup is full even if the camera pans away and returns. It's the foundation for high-level spatial reasoning and robotic interaction.

*Engineering the future, one frame at a time.*
🔧🧪
