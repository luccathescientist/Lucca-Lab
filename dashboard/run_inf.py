import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

def run_inference(model_id, prompt):
    if model_id == "deepseek-70b-fp8":
        # Use persistent vLLM server on port 8001
        import requests
        url = "http://localhost:8001/v1/completions"
        payload = {
            "model": "neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic",
            "prompt": prompt,
            "max_tokens": 512,
            "stream": True
        }
        response = requests.post(url, json=payload, stream=True)
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode('utf-8').replace('data: ', ''))
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    text = chunk['choices'][0].get('text', '')
                    sys.stdout.write(text)
                    sys.stdout.flush()
        return

    if model_id == "deepseek-70b":
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        load_args = {"device_map": "auto", "load_in_4bit": True}
    elif model_id == "deepseek-8b":
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-8B"
        load_args = {"device_map": "auto", "torch_dtype": torch.float16}
    else:
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        load_args = {"device_map": "auto", "torch_dtype": torch.float16}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_args)

    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=512, do_sample=True, temperature=0.7)
    
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
