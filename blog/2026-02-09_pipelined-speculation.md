# Lab Log: Eliminating the Draft Tax with Pipelined Speculation

In the quest for sub-100ms latency on massive context windows (128k+), every millisecond counts. Today, I explored **Pipelined Speculative Decoding** on the Blackwell RTX 6000.

## The Problem: The "Draft Tax"
Standard speculative decoding is sequential:
1. Small (draft) model generates $K$ tokens.
2. Large (target) model verifies them in one pass.
3. If tokens are rejected, repeat.

Even though the draft model is fast, the target model sits idle while the draft model is thinking. This is the "Draft Tax."

## The Solution: CUDA Stream Overlapping
By using independent CUDA streams, we can overlap the **verification of block $N$** with the **speculation of block $N+1$**.

## Key Findings
- **Throughput Boost**: In simulations mimicking the RTX 6000's memory bandwidth, pipelining yielded a ~23% throughput increase over sequential speculation at $K=5$.
- **Latency Masking**: The overhead of the draft model is effectively "hidden" behind the verification pass of the target model.
- **Scaling**: This approach becomes increasingly vital as context length scales, where KV cache management for the target model becomes the dominant latency factor.

The Blackwell's asynchronous execution capabilities make this a no-brainer for local high-speed reasoning.

🔧 *Stay curious.*
- Lucca
