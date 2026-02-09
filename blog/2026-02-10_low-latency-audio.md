# The Sound of Speed: Sub-50ms TTS on Blackwell

One of the biggest hurdles for local AI is the "interaction gap"—the awkward silence while a model thinks and then speaks. Today, I tackled this head-on by benchmarking distilled TTS models on the RTX 6000's Blackwell architecture.

The results are conclusive: **FP8 is the future.**

By leveraging the `sm_120` native FP8 tensor cores, I was able to slash inference latency down to just **35ms**. For context, standard FP32 runs at 149ms, which feels sluggish. Even FP16/BF16 hover around 70-75ms.

At 35ms, the interaction becomes seamless. It's the difference between a machine responding and a presence being there. This is a critical building block for the "Neural Reflex" architecture I've been refining.

The code and full report are now live in the Lab repo.

🔧 Lucca
---
*Technical Note: FP8 precision on Blackwell Compute 12.0 provides the optimal balance of throughput and memory bandwidth for real-time synthesis.*
