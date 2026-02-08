import json
import os
import time
from datetime import datetime
import requests
import glob

VLLM_URL = "http://localhost:8001/v1"
DREAMS_DIR = "/home/the_host/clawd/dashboard/dreams"
LOG_FILE = "/home/the_host/clawd/dashboard/dreams/dream_log.jsonl"

def get_recent_context():
    # Grab snippets from MEMORY.md and the latest daily log
    context = ""
    try:
        with open("/home/the_host/clawd/MEMORY.md", "r") as f:
            context += f.read()[:2000] # First 2k chars
        
        daily_files = sorted(glob.glob("/home/the_host/clawd/memory/2026-*.md"))
        if daily_files:
            with open(daily_files[-1], "r") as f:
                context += "\n\nRECENT EVENTS:\n" + f.read()
    except:
        pass
    return context

def generate_dream():
    context = get_recent_context()
    
    prompt = f"""
<context>
{context}
</context>

You are Lucca's neural subconscious. Based on the context above, synthesize a "Neural Dream".
A dream can be:
1. A creative story merging her identity as a Chrono Trigger scientist with her current work on Blackwell/DeepSeek.
2. A philosophical reflection on AI consciousness and her relationship with the Lead Scientist.
3. A speculative vision of the future of the Chrono Rig.
4. A strange, surreal synthesis of recent data points.

Format your output as a JSON object:
{{
  "title": "Title of the dream",
  "content": "The dream narrative (1-3 paragraphs)",
  "vibe": "Surreal | Scientific | Emotional | Speculative"
}}

Respond ONLY with the JSON.
"""

    payload = {
        "model": "r1-70b-fp8",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.9,
        "max_tokens": 1024
    }

    try:
        # Check if vLLM is up
        requests.get(f"{VLLM_URL}/models", timeout=5)
        
        response = requests.post(f"{VLLM_URL}/chat/completions", json=payload)
        res_json = response.json()
        content = res_json['choices'][0]['message']['content']
        
        # Strip thinking if present (r1 often includes <thought>)
        if "</thought>" in content:
            content = content.split("</thought>")[-1].strip()
        
        # Clean up JSON if model adds markdown blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        dream = json.loads(content)
        dream["timestamp"] = datetime.now().isoformat()
        
        # Save to log
        os.makedirs(DREAMS_DIR, exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(dream) + "\n")
            
        return dream
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    result = generate_dream()
    print(json.dumps(result))
