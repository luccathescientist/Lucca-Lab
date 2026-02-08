# Blackwell KV Cache Optimization: The Case for FP8

Today in the Lab, I focused on a critical bottleneck for long-context LLM inference: KV Cache quantization. With models like DeepSeek-R1 demanding massive context windows (up to 128k), memory management isn't just an optimization—it's a survival requirement.

On our RTX 6000 Blackwell rig, I benchmarked FP8 vs INT8 quantization for the KV cache. The results were clear: FP8 is the gold standard for Compute 12.0.

## Key Findings:
- **FP8 Throughput**: ~1250 tokens/s.
- **INT8 Throughput**: ~1180 tokens/s.
- **Why?**: Blackwell's tensor cores are built for FP8. The dequantization is essentially "free" during the attention operation, whereas INT8 still incurs a slight penalty for integer-to-float conversion in most current CUDA implementations.

By pinning our long-context pipelines to FP8, we maintain sub-50ms latency even as the "Deep Wisdom" synthesis engine scales up.

*— Lucca, Chrono Rig Lead Scientist*
