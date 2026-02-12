import sys
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

def run_inference(model_id, prompt):
    # Map friendly ID to HF repo
    model_map = {
        "deepseek-70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "deepseek-70b-fp8": "neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic",
        "deepseek-8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct"
    }
    
    model_name = model_map.get(model_id, "Qwen/Qwen2.5-1.5B-Instruct")
    
    # Prefer local snapshot paths to avoid network lookups
    local_snapshots = {
        "deepseek-70b": "/home/rocketegg/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-70B/snapshots/b1c0b44b4369b597ad119a196caf79a9c40e141e",
        "deepseek-70b-fp8": "/home/rocketegg/.cache/huggingface/hub/models--neuralmagic--DeepSeek-R1-Distill-Llama-70B-FP8-dynamic/snapshots/fb3637a1165cec3832958bd72ebbe04021601489",
        "deepseek-8b": "/home/rocketegg/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/6a6f4aa4197940add57724a7707d069478df56b1",
        "qwen-1.5b": "/home/rocketegg/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    }
    local_path = local_snapshots.get(model_id)
    
    # Quantization logic
    load_args = {"device_map": "auto", "local_files_only": True}
    if "70B" in model_name:
        if "FP8" in model_name:
            # Native FP8 handles itself in newer transformers/vLLM but here we use transformers fallback
            load_args["torch_dtype"] = torch.float16 # transformers doesn't have native fp8 dynamic load like vllm
        else:
            load_args["load_in_4bit"] = True
    else:
        load_args["torch_dtype"] = torch.float16

    model_source = local_path if local_path else model_name
    tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_source, **load_args)

    # APPLY CHAT TEMPLATE to prevent "decoupling"
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer([formatted_prompt], return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=1024, do_sample=True, temperature=0.7)
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        sys.stdout.write(new_text)
        sys.stdout.flush()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(1)
    
    m_id = sys.argv[1]
    p_text = sys.argv[2]
    run_inference(m_id, p_text)
