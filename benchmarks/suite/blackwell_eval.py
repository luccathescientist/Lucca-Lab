
import os
import sys
import time
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import psutil

def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)
    return 0

def benchmark_model(model_id, prompt="Explain the significance of the Blackwell architecture in AI computing."):
    print(f"--- Benchmarking: {model_id} ---")
    
    start_load = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
    except Exception as e:
        return {"error": str(e)}
    
    load_time = time.time() - start_load
    vram_after_load = get_gpu_memory()
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Warmup
    _ = model.generate(**inputs, max_new_tokens=5)
    
    start_gen = time.time()
    outputs = model.generate(**inputs, max_new_tokens=100)
    gen_time = time.time() - start_gen
    
    num_tokens = len(outputs[0]) - len(inputs["input_ids"][0])
    tokens_per_sec = num_tokens / gen_time
    
    report = {
        "model": model_id,
        "load_time_sec": round(load_time, 2),
        "inference_time_sec": round(gen_time, 2),
        "tokens_generated": num_tokens,
        "tokens_per_sec": round(tokens_per_sec, 2),
        "vram_usage_gb": round(vram_after_load, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return report

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python blackwell_eval.py <model_id>")
        sys.exit(1)
        
    model_name = sys.argv[1]
    result = benchmark_model(model_name)
    
    output_file = f"report_{model_name.replace('/', '_')}.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)
        
    print(json.dumps(result, indent=4))
