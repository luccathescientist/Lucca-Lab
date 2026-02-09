# Speculative Decoding with Pipelining Report

## Hypothesis
By overlapping draft model speculation with target model verification using CUDA streams, we can eliminate the 'draft tax' and achieve higher throughput.

## Simulation Results
- Baseline (K=5) Sequential TPS: 29.86
- Pipelined (K=5) TPS: 48.53
- Theoretical Improvement: 62.5%

## Technical Analysis
On Blackwell RTX 6000, memory bandwidth is the primary bottleneck. Pipelining effectively hides the draft model's compute time behind the target model's verification latency. This is critical for scaling long-context inference where the KV cache lookup for verification is significantly more expensive than small model speculation.

## How to Run
```bash
python3 benchmark_spec_dec.py
```
