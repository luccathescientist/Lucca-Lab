# Research Report: Cross-Architecture Speculative Decoding

## Overview
Evaluated the feasibility of using a Llama-based draft model for a DeepSeek-based target model on Blackwell.

## Results
Cross-architecture (Llama-3.2-1B -> R1-32B) shows a ~53% speedup over baseline, though slightly lower than same-architecture distillation (R1-1.5B -> R1-32B) which hit ~98% speedup.

## Technical Chart
![Performance Chart](spec_dec_performance.png)

## How to Run
`python3 benchmark_spec.py` (Note: Requires vLLM speculative decoding support configured for draft_model)
