import time
from vllm import LLM, SamplingParams

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
prompts = ["Hello, who are you?", "Explain the Blackwell architecture."]
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=100)

print(f"Loading small model: {model_name}...")
llm = LLM(model=model_name)

print("Generating...")
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt!r}")
    print(f"Generated text: {output.outputs[0].text!r}")
    print("-" * 20)
