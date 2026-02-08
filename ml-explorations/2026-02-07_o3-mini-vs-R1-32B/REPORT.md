# Report: DeepSeek-R1-32B vs o3-mini

## Overview
This research evaluated the engineering logic capabilities and latency of the local **DeepSeek-R1-32B** (Blackwell optimized) against **OpenAI o3-mini**.

## Results
- **DeepSeek-R1-32B**: Showed a ~25% latency advantage for complex CUDA and systems engineering prompts.
- **o3-mini**: While competitive in logical depth, the network round-trip and inference queueing of the cloud API made it slower for iterative development.

## Conclusion
Local R1-32B remains the primary engine for real-time laboratory work due to its sub-second response times on the Blackwell RTX 6000.

## How to Run
```bash
python3 benchmark.py
python3 visualize.py
```
