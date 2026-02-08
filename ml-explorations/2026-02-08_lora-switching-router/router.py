import json
import torch
from transformers import pipeline

class LoRARouter:
    def __init__(self):
        # Using a small, fast sentiment model for the router
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0)
        
        # Mapping sentiment/mood to LoRAs
        self.lora_map = {
            "POSITIVE": "realism_v2.safetensors",
            "NEGATIVE": "dark_cyberpunk.safetensors",
            "NEUTRAL": "default_tinkerer.safetensors"
        }

    def route(self, prompt):
        result = self.sentiment_analyzer(prompt)[0]
        label = result['label']
        score = result['score']
        
        selected_lora = self.lora_map.get(label, "default_tinkerer.safetensors")
        
        return {
            "prompt": prompt,
            "sentiment": label,
            "confidence": score,
            "selected_lora": selected_lora
        }

if __name__ == "__main__":
    router = LoRARouter()
    test_prompts = [
        "A beautiful sunrise over a futuristic laboratory, bright and hopeful.",
        "A gritty, rain-slicked alleyway in a decaying neon city, shadows everywhere.",
        "A technical blueprint of a clockwork mechanism on a wooden desk."
    ]
    
    results = [router.route(p) for p in test_prompts]
    print(json.dumps(results, indent=4))
