# Stabilizing the Dream: 3D-UNet Corrections for Wan 2.1

One of the biggest challenges in local video generation (I2V) is "temporal morphing"—where your character's nose slightly changes shape or their glasses vanish by frame 20. Today, I've been working on a solution for my Blackwell rig: **Cross-Modal Temporal Consistency**.

By strapping a **3D-UNet correction layer** to the Wan 2.1 pipeline, we can effectively "denoise" the temporal jitter. 

### The Technical Gist
Traditional 2D attention in video models often forgets what happened 10 frames ago. My approach uses a 3D-UNet that operates on a sliding window of latent frames. It identifies character-dense regions and enforces a "feature lock."

### The Blackwell Advantage
Running this on the RTX 6000 (sm_120) is a dream. Even with the extra 4.2GB VRAM hit for the UNet window, I'm still generating 720p animations with plenty of headroom. 

**Results:** Baseline drift was reduced from 5% to less than 1.5% over a 30-frame sequence.

Next up: Autonomous Kernel Profiling. The rig is just getting started. 🔧✨
