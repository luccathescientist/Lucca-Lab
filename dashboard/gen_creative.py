import json
import os
import requests
import random
import time
import sys

COMFY_URL = "http://127.0.0.1:8188"

def generate_creative(lora_name, prompt, filename, base_model=None):
    workflow_path = "/home/rocketegg/clawd/workflows/workflow_lora_detail.json"
    with open(workflow_path, "r") as f:
        workflow = json.load(f)
    
    # Update prompt
    workflow["6"]["inputs"]["text"] = prompt
    # Update base model
    if base_model:
        workflow["4"]["inputs"]["ckpt_name"] = base_model
    # Update LoRA
    if lora_name:
        workflow["12"]["inputs"]["lora_name"] = lora_name
    # Random seed
    workflow["3"]["inputs"]["seed"] = random.randint(0, 10**16)
    
    p = {"prompt": workflow}
    data = json.dumps(p).encode('utf-8')
    
    try:
        response = requests.post(f"{COMFY_URL}/prompt", data=data, timeout=5)
        if response.status_code != 200:
            print(f"Error: ComfyUI returned {response.status_code}: {response.text}")
            return
        
        prompt_id = response.json().get('prompt_id')
        if not prompt_id:
            print("Error: Missing prompt_id from ComfyUI")
            return
        print(f"Generation started: {prompt_id}")
        
        start = time.time()
        while True:
            if time.time() - start > 180:
                print("Error: Generation timed out")
                return
            history_resp = requests.get(f"{COMFY_URL}/history/{prompt_id}")
            history = history_resp.json()
            if prompt_id in history:
                image_info = history[prompt_id]['outputs']['9']['images'][0]
                image_data = requests.get(f"{COMFY_URL}/view?filename={image_info['filename']}&subfolder={image_info['subfolder']}&type={image_info['type']}").content
                with open(filename, "wb") as f:
                    f.write(image_data)
                print(f"Image saved: {filename}")
                break
            time.sleep(2)
    except Exception as e:
        print(f"Generation failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(1)
    
    l_name = sys.argv[1]
    p_text = sys.argv[2]
    out_file = sys.argv[3]
    base_model = sys.argv[4] if len(sys.argv) > 4 else None
    generate_creative(l_name, p_text, out_file, base_model)
