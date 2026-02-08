import time
import json
import os

# Evaluation questions for logic, math, and code
EVAL_SET = [
    {
        "id": 1,
        "category": "Math/Logic",
        "question": "If I have 3 oranges and you have 2 apples, and we swap one orange for an apple, how many fruits do I have total? Explain the reasoning step-by-step."
    },
    {
        "id": 2,
        "category": "Code/Engineering",
        "question": "Write a highly optimized Python function to find the first 1000 prime numbers. Then, explain the time complexity using Big O notation."
    },
    {
        "id": 3,
        "category": "Causal Inference",
        "question": "A room has 3 light switches outside, all off. They control 3 bulbs inside. You can go inside once. How do you determine which switch controls which bulb?"
    }
]

def log_result(model_name, task_id, prompt, response, latency):
    log_file = f"eval_results_{model_name}.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps({
            "model": model_name,
            "task_id": task_id,
            "prompt": prompt,
            "response": response,
            "latency": latency,
            "timestamp": time.time()
        }) + "\n")

if __name__ == "__main__":
    print("Evaluation Script Initialized.")
    # This script acts as a template/placeholder for the manual triggering done by Lucca
    # In a real scenario, this would call the respective model APIs.
