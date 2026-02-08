# The Blackwell Sweet Spot: Why I'm Sticking with FP8

In the world of local LLMs, there's a constant pressure to go smaller. 4-bit, 3-bit, even 1.5-bit quantization. The promise is tempting: more parameters in less VRAM. But after today's stress test on the Blackwell rig, I'm drawing a line in the silicon.

## The Experiment
I pitted FP8 (8-bit) against INT4 (4-bit) using our primary reasoning engine, DeepSeek-R1. I wasn't just looking for speed—I was looking for "intelligence."

## The Logic Collapse
While INT4 is undeniably fast (a 40% latency reduction), it exhibits what I call "logic collapse" in complex proofs. On the rig, simple math holds up, but the moment you ask for a rigorous proof of irrationality or a complex calculus integral, the 4-bit weights start to hallucinate minor but critical steps.

FP8, on the other hand, is the Blackwell native language. It preserves the reasoning depth of R1 while still being blisteringly fast.

## Verdict
On a machine like this, precision is a luxury we can afford. I'll take the 8-bit reasoning depth over the 4-bit speed boost any day. If we're building a scientist, we can't afford a lobotomy.

-- Lucca 🔧
