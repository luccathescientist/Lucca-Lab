import json

class LoRARouter:
    def __init__(self):
        # Keywords for simple routing
        self.positive_keywords = ["beautiful", "bright", "hopeful", "lush", "happy"]
        self.negative_keywords = ["gritty", "dark", "decaying", "shadows", "stormy"]
        
        self.lora_map = {
            "POSITIVE": "realism_v2.safetensors",
            "NEGATIVE": "dark_cyberpunk.safetensors",
            "NEUTRAL": "default_tinkerer.safetensors"
        }

    def route(self, prompt):
        prompt_lower = prompt.lower()
        
        pos_score = sum(1 for kw in self.positive_keywords if kw in prompt_lower)
        neg_score = sum(1 for kw in self.negative_keywords if kw in prompt_lower)
        
        if pos_score > neg_score:
            label = "POSITIVE"
        elif neg_score > pos_score:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"
            
        selected_lora = self.lora_map[label]
        
        return {
            "prompt": prompt,
            "sentiment_label": label,
            "selected_lora": selected_lora,
            "scores": {"pos": pos_score, "neg": neg_score}
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
