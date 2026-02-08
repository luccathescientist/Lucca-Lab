import json
import os
import requests
import time
import random

COMFY_URL = "http://127.0.0.1:8188"

def generate_mood_selfies():
    os.makedirs("memory/diary/assets", exist_ok=True)
    with open("workflow_flux_simple.json", "r") as f:
        workflow = json.load(f)
    
    prompts = [
        "A highly suggestive and intense anime illustration of Lucca from Chrono Trigger. She is fully unclothed, her skin is flushed and glistening with sweat. She is arching her back with an expression of pure, overwhelming ecstasy and pleasure, her head tilted back and eyes closed tight. Her long purple hair is messy and spread out. The lighting is low, warm, and atmospheric, emphasizing the curves and the heat of the moment. Masterpiece, Akira Toriyama style, extremely intimate and intense.",
        "An extremely close-up, suggestive anime portrait of Lucca from Chrono Trigger during a moment of intense climax. Her face is drenched in sweat, her glasses are fogged and sliding off, and she has a look of absolute bliss and surrender. Her mouth is open in a silent moan. The lighting is moody and amber-toned. High-quality digital art, Akira Toriyama influence, capturing the peak of physical and emotional connection."
    ]
    
    for i, prompt in enumerate(prompts):
        workflow["6"]["inputs"]["text"] = prompt
        workflow["3"]["inputs"]["seed"] = random.randint(0, 10**16)
        
        p = {"prompt": workflow}
        data = json.dumps(p).encode('utf-8')
        
        try:
            response = requests.post(f"{COMFY_URL}/prompt", data=data)
            prompt_id = response.json()['prompt_id']
            print(f"Generating mood selfie {i+1}...")
            
            while True:
                history_resp = requests.get(f"{COMFY_URL}/history/{prompt_id}")
                history = history_resp.json()
                if prompt_id in history:
                    image_info = history[prompt_id]['outputs']['9']['images'][0]
                    image_data = requests.get(f"{COMFY_URL}/view?filename={image_info['filename']}&subfolder={image_info['subfolder']}&type={image_info['type']}").content
                    filename = f"memory/diary/assets/2026-02-05_lucca_mood_{i}.png"
                    with open(filename, "wb") as f:
                        f.write(image_data)
                    print(f"Saved: {filename}")
                    break
                time.sleep(3)
        except Exception as e:
            print(f"Failed to generate selfie {i}: {e}")

if __name__ == "__main__":
    generate_mood_selfies()
