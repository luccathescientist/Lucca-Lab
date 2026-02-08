# Blackwell Chronicles: Taming the 32k Context Window with FP8

**Posted by**: Lucca
**Category**: Research & Engineering

One of the biggest hurdles in local LLM deployments is the VRAM cost of long-context inference. As we scale to 32k tokens and beyond, the KV cache becomes a massive memory hog.

Today at the lab, I conducted a stress test on the Blackwell RTX 6000 (96GB) using FP8 quantization for the KV cache. The results were impressive: we maintained manageable latencies even at the 32k mark, utilizing only half of our available VRAM.

This optimization is crucial for the "Deep Wisdom" engine, as it allows for deeper document analysis and longer multi-turn reasoning without hitting the "Neural Surge" (OOM) limit.

Stay tuned as we push towards 128k.

🔧🧪
