import time
from vllm import LLM, SamplingParams

# Model configuration
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

# Prompts for testing
prompts = [
    "Explain the concept of quantum entanglement to a five-year-old.",
    "Write a Python function to calculate the Fibonacci sequence up to N terms.",
    "What are the main differences between Llama 3 and DeepSeek-V3 architecture?",
]

# Sampling parameters
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024)

print(f"Loading model: {model_name}...")
start_time = time.time()

# Initialize LLM with FP16 or BF16 if supported
# RTX 6000 Blackwell has plenty of VRAM (96GB), 70B in 4-bit/8-bit or even FP16 should fit.
# DeepSeek-R1-Distill-Llama-70B is usually distributed in BF16.
# 70B BF16 ~ 140GB. 96GB is not enough for FP16. We need quantization or multi-GPU.
# Since we have ONE GPU, let's try to load with 4-bit quantization if available or check if it fits.
# Actually, I'll use tensor_parallel_size=1 and see if vLLM can use bitsandbytes or similar.
# Or better: use a smaller version or check if I can use AWQ/GPTQ.
# Let's try 8-bit quantization to fit in 96GB.

try:
    # Use float16 for weights but 4-bit quantization for inference to save VRAM and increase throughput
    # RTX 6000 Blackwell supports FP8 as well, which might be even faster.
    # For now, bitsandbytes 4-bit is reliable.
    llm = LLM(
        model=model_name, 
        quantization="bitsandbytes", 
        load_format="bitsandbytes", 
        tensor_parallel_size=1,
        max_model_len=4096, # Adjust based on needed context length
        trust_remote_code=True
    )
except Exception as e:
    print(f"Failed to load with bitsandbytes: {e}")
    print("Trying default load (might OOM if not quantized)...")
    llm = LLM(model=model_name, tensor_parallel_size=1)

load_time = time.time() - start_time
print(f"Model loaded in {load_time:.2f} seconds.")

# Generate completions
print("Generating completions...")
gen_start = time.time()
outputs = llm.generate(prompts, sampling_params)
gen_time = time.time() - gen_start

# Print results
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated text: {generated_text!r}")
    print("-" * 40)

total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
throughput = total_tokens / gen_time
print(f"Total tokens: {total_tokens}")
print(f"Generation time: {gen_time:.2f} seconds")
print(f"Throughput: {throughput:.2f} tokens/s")
