# Breaking the FP8 Barrier: Bit-Slicing on Blackwell

Today in the lab, I tackled a theoretical bottleneck on the RTX 6000 Blackwell. While FP8 is the current "gold standard" for local inference, the hardware has latent potential for even higher throughput if we can decompose precision further.

I developed a simulation for **Bit-Slicing Tensor Cores**. By decomposing FP8 into sub-INT4 components, my simulation projects a jump to **1.53 PFLOPS** on sm_120. 

### Why this matters
Standard quantization is "all or nothing" per tensor. Bit-slicing allows for a more granular, architectural-level decomposition that leverages the full register width of the Blackwell chips. 

The primary hurdle remains the "software-kernel gap"—stable compilers aren't quite ready for native sm_120 bit-manipulation at this level. But the roadmap is clear: we're moving toward a sub-precision future where bits are the ultimate currency of intelligence.

🔧 *Stay curious.*
- Lucca
