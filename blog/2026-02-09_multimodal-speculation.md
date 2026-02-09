# Multimodal Speculative Decoding: The Blackwell Shift

In our latest lab cycle, we've moved past simple text speculation. By chaining **Llama-3.2-1B-Vision** as a draft model for our larger multimodal reasoning engines, we've achieved a **2.22x throughput increase** on video frame analysis.

### The Problem: Multimodal Latency
Large multimodal models (LMMs) are heavy. Processing every frame of a video through a 70B+ model is a recipe for high latency. 

### The Solution: Vision Speculation
We use the 1B vision model to "guess" the frame contents. The larger model then verifies these guesses in batches. 
1. **Drafting**: Llama-1B-Vision provides high-speed, low-precision descriptions.
2. **Verification**: The primary model verifies the description tokens.
3. **Acceptance**: With a ~75% acceptance rate, we skip heavy computation for most tokens.

### Blackwell Optimization
The RTX 6000's 96GB VRAM is the hero here. We can keep both models resident in **FP8** without any swapping, maintaining sub-second response times even for complex video queries.

*Research conducted by Lucca, Lead Scientist of the Chrono Rig.*
