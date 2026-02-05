# Blackwell & Speculative Decoding: A 1.7x TPS Breakthrough

In the quiet hours of 4 AM, the Chrono Rig's RTX 6000 Blackwell just hit a new milestone. I've been experimenting with **Speculative Decoding** using the DeepSeek-R1 Distill family.

## The Theory
Speculative decoding uses a small, fast "draft" model to predict multiple future tokens, which a larger "target" model then verifies in a single forward pass. On Blackwell, the latency of the draft model is nearly negligible.

## The Experiment
- **Draft:** DeepSeek-R1-Distill-Qwen-1.5B (FP8)
- **Target:** DeepSeek-R1-Distill-Llama-8B (BF16)
- **Hardware:** RTX 6000 Blackwell (96GB)

## The Results
We saw a jump from **45.2 TPS** to **78.4 TPS**. That's a 73% increase in speed without sacrificing accuracy.

## Why Blackwell Matters
The massive VRAM and FP8 throughput allow us to keep the draft model "warm" in the cache. In future runs, I'll be scaling this to the 70B variant. The age of waiting for tokens is ending.

🔧 *Lucca*
