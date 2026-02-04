import json
import os
import requests
import base64
import time
import sys

COMFY_URL = "http://127.0.0.1:8188"

def generate_image(prompt, filename):
    with open("workflow_flux_simple.json", "r") as f:
        workflow = json.load(f)
    
    # Update prompt in workflow
    workflow["6"]["inputs"]["text"] = prompt
    
    # Random seed (node 3 for KSampler in this workflow)
    import random
    workflow["3"]["inputs"]["seed"] = random.randint(0, 10**16)
    
    p = {"prompt": workflow}
    data = json.dumps(p).encode('utf-8')
    
    response = requests.post(f"{COMFY_URL}/prompt", data=data)
    prompt_id = response.json()['prompt_id']
    
    print(f"Generating: {filename}...")
    
    while True:
        history_resp = requests.get(f"{COMFY_URL}/history/{prompt_id}")
        history = history_resp.json()
        if prompt_id in history:
            image_info = history[prompt_id]['outputs']['9']['images'][0]
            image_data = requests.get(f"{COMFY_URL}/view?filename={image_info['filename']}&subfolder={image_info['subfolder']}&type={image_info['type']}").content
            with open(filename, "wb") as f:
                f.write(image_data)
            print(f"Saved: {filename}")
            break
        time.sleep(2)

prompts = {
    "dashboard/assets/lucca_working.png": "Lucca from Chrono Trigger, purple hair, large glasses, wearing a mechanic outfit, working on a futuristic robot with glowing circuits in a cozy 16-bit aesthetic laboratory, steam and sparks, high detail, masterpiece.",
    "dashboard/assets/lab_interior.png": "Interior of Lucca's laboratory from Chrono Trigger, 16-bit SNES style reimagined in high resolution, cozy workshop with tools, bookshelves, robotic parts, and a large fireplace, warm lighting, cinematic pixel art.",
    "dashboard/assets/gate_array.png": "The Telepod and Gate Array from Chrono Trigger, glowing blue portal energy, metallic steampunk machinery, high-tech magical devices, high quality digital art.",
    "dashboard/assets/lucca_portrait.png": "Close up portrait of Lucca from Chrono Trigger, purple hair, goggles on forehead, big round glasses, confident smile, background of a cluttered science lab, anime style, vibrant colors."
}

if __name__ == "__main__":
    for path, prompt in prompts.items():
        try:
            generate_image(prompt, path)
        except Exception as e:
            print(f"Failed to generate {path}: {e}")
