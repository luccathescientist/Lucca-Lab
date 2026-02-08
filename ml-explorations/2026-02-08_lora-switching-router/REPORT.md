# Report: Automated LoRA Switching Router

## Overview
This project implements a sentiment-based router for dynamically selecting Flux LoRAs based on prompt content. The goal is to automate the selection of the most appropriate aesthetic LoRA without manual user intervention.

## Implementation
- **Router Logic**: A lightweight keyword-based sentiment analyzer (expandable to LLM-based analysis).
- **Mapping**:
    - **Positive**: `realism_v2.safetensors`
    - **Negative**: `dark_cyberpunk.safetensors`
    - **Neutral**: `default_tinkerer.safetensors`

## Results
The router successfully classified test prompts:
1. "Beautiful sunrise..." -> **POSITIVE** -> `realism_v2.safetensors`
2. "Gritty alleyway..." -> **NEGATIVE** -> `dark_cyberpunk.safetensors`
3. "Technical blueprint..." -> **NEUTRAL** -> `default_tinkerer.safetensors`

## How to Run
```bash
python3 router_light.py
```

## Future Work
- Integrate with `vLLM` to use `DeepSeek-R1-32B` for complex semantic sentiment analysis.
- Implement the actual filesystem swap/API call to the image generation backend.
