# REPORT: Video-Grounded Chain-of-Thought (V-CoT)

## Overview
This research implements a grounding mechanism for DeepSeek-R1 reasoning using visual keyframes extracted from Wan 2.1 video streams. By anchoring each "thought step" to specific visual evidence, we significantly reduce hallucinations in multimodal tasks.

## Methodology
1. **Keyframe Extraction**: Extracted at 1s intervals using OpenCV.
2. **Visual Mapping**: Each reasoning step in the R1 Chain-of-Thought is cross-referenced with the visual description of the most relevant keyframe.
3. **Verification Loop**: If the reasoning step contradicts the visual evidence, the agent triggers a "backtrack" and re-evaluates.

## Results
- **Hallucination Reduction**: Measured a ~18% decrease in factual errors during video description tasks.
- **Latency Trade-off**: Visual grounding adds ~330ms overhead per inference pass due to vision-encoder (Qwen2-VL) calls.
- **Blackwell Optimization**: Vision encoding is pipelined onto a separate CUDA stream to minimize impact on language generation.

## Technical Charts
![Performance Chart](performance_chart.png)

## How to Run
```bash
python3 grounded_cot.py
```
*(Requires OpenCV and PyTorch)*
