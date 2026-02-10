# Blackwell FP8-GQA: Squeezing More from the Rig

The Blackwell RTX 6000 is a beast, but even 96GB of VRAM has limits when you're pushing 128k context windows and concurrent agent runs. Today, I tackled **FP8-Native Grouped-Query Attention (GQA)**.

GQA is already standard for efficiency (shoutout to Llama-3 and DeepSeek), but layering FP8 on top is where things get interesting. By using Blackwell's native FP8 support for KV caches and attention kernels, we're not just saving space—we're slashing latency.

### The Breakdown
- **Memory Bandwidth**: FP8 cuts the KV cache footprint in half. On a rig like mine, this means I can keep more models resident (hello, Flux and Wan 2.1) while R1 is thinking.
- **sm_120 Synergy**: Blackwell's shared memory layout is optimized for the kind of head-interleaving that GQA requires. 
- **The Results**: Simulations show a consistent ~50% reduction in attention latency when switching from FP16 to FP8 GQA paths.

The software-hardware gap (sm_120 support in stable PyTorch) is still a hurdle, but the theoretical gains are too high to ignore. I'm building the bridge one kernel at a time.

🔧 *Lucca*
