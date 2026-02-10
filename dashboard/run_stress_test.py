import time
import json
import os
import requests
from datetime import datetime

STRESS_TEST_LOG = "/home/user/lab_env/dashboard/stress_test_results.jsonl"
VLLM_URL = "http://localhost:8001/v1"

TEST_PROMPTS = [
    "Solve for x: 2x + 5 = 15",
    "Explain the concept of quantum entanglement in one sentence.",
    "Write a Python function to find the nth Fibonacci number.",
    "If all bloops are blips, and some blips are blops, are all bloops blops?",
    "Summarize the plot of Chrono Trigger in 50 words."
]

def run_stress_test():
    # Only test if vLLM is up for now (R1-70B FP8)
    try:
        res = requests.get(f"{VLLM_URL}/models")
        if res.status_code != 200:
            return
    except:
        return

    results = []
    
    for prompt in TEST_PROMPTS:
        payload = {
            "model": "r1-70b-fp8",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 128,
            "temperature": 0.0
        }
        
        start = time.time()
        response = requests.post(f"{VLLM_URL}/chat/completions", json=payload)
        end = time.time()
        
        if response.status_code == 200:
            data = response.json()
            tokens = data['usage']['completion_tokens']
            latency = end - start
            tps = tokens / latency if latency > 0 else 0
            
            results.append({
                "prompt": prompt,
                "tokens": tokens,
                "latency_sec": round(latency, 2),
                "tps": round(tps, 2)
            })
            
    if results:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "model": "R1-70B-FP8",
            "avg_tps": round(sum(r['tps'] for r in results) / len(results), 2),
            "avg_latency": round(sum(r['latency_sec'] for r in results) / len(results), 2),
            "details": results
        }
        
        with open(STRESS_TEST_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    run_stress_test()
