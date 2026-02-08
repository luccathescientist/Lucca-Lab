# Exploration Report: DeepSeek-R1-Distill-Llama-70B with vLLM

## Objective
To benchmark and explore the performance of the DeepSeek-R1-Distill-Llama-70B model using the vLLM inference engine on an RTX PRO 6000 Blackwell GPU.

## Setup
- **Model**: `deepseek-ai/DeepSeek-R1-Distill-Llama-70B`
- **Engine**: vLLM v0.15.0
- **Hardware**: NVIDIA RTX PRO 6000 Blackwell (96GB VRAM)
- **Quantization**: 4-bit (via `bitsandbytes` in vLLM)
- **Environment**: Custom Python 3.10 venv

## Results
- **Weight Loading**: ~56 seconds.
- **Memory Footprint**: ~40GB (Weights) + ~40GB (KV Cache) = ~80GB total VRAM usage.
- **Throughput**: **14.87 tokens/s** (Excellent for 70B model on single GPU).
- **Generation Quality**: 
  - **Prompt 1 (Quantum Entanglement)**: Successful reasoning and explanation.
  - **Prompt 2 (Fibonacci Python)**: Correct implementation with detailed reasoning trace.
  - **Prompt 3 (Llama 3 vs DeepSeek-V3)**: The model was conservative, stating it doesn't have access to "internal company information," likely a safety or knowledge cutoff behavior.

## Summary
The RTX PRO 6000 Blackwell (96GB VRAM) easily handles the DeepSeek-R1-Distill-Llama-70B model using 4-bit quantization. The vLLM engine with `bitsandbytes` and `torch.compile` provides high throughput (14.87 tokens/s), making it highly viable for local development and reasoning tasks.

## Challenges
- **Download Size**: 140GB of weights took significant time to download and verify.
- **Dependency Conflicts**: Custom modeling code for DeepSeek-OCR was incompatible with the latest `transformers` library (missing `LlamaFlashAttention2`).

## Next Steps
- Experiment with FP8 quantization for even higher throughput on Blackwell.
- Fix the OCR model dependencies by aligning custom modeling code with `transformers` v4.57+.
- Test larger context lengths (8k+).
