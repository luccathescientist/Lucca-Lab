
import os
import sys
import time
import json
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add local site packages
sys.path.append(os.path.abspath('./local_site'))

QA_TESTS = [
    {"q": "If I have 5 apples and buy 3 more, then give 2 to a friend, how many do I have?", "a": "6"},
    {"q": "What is the square root of 144?", "a": "12"},
    {"q": "If a train travels at 60mph for 3 hours, how far does it go?", "a": "180"},
    {"q": "Solve for x: 2x + 10 = 20", "a": "5"},
    {"q": "A shirt costs $20 and is on sale for 25% off. What is the final price?", "a": "15"}
]

def extract_answer(text):
    # Try \boxed{} first
    boxed = re.findall(r'\\boxed\{(\d+)\}', text)
    if boxed: return boxed[-1]
    
    # Try "Final Answer: <number>"
    final = re.findall(r'Final Answer:\s*(\d+)', text, re.IGNORECASE)
    if final: return final[-1]
    
    # Try any number at the very end
    numbers = re.findall(r'(\d+)', text)
    if numbers: return numbers[-1]
    
    return ""

def run_quality_test(model_id):
    print(f"--- Quality Testing: {model_id} ---")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    correct = 0
    results = []
    
    for test in QA_TESTS:
        prompt = f"Question: {test['q']}\nReason step by step and provide the final numeric answer in \\boxed{{}}."
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=300)
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        received_answer = extract_answer(full_response)
        
        is_correct = test['a'] == received_answer
        if is_correct: correct += 1
        
        results.append({
            "question": test['q'],
            "expected": test['a'],
            "received": received_answer,
            "correct": is_correct
        })
        print(f"Q: {test['q']} | R: {received_answer} | {'✅' if is_correct else '❌'}")

    score = (correct / len(QA_TESTS)) * 100
    report = {
        "model": model_id,
        "accuracy_score": score,
        "details": results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return report

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quality_eval.py <model_id>")
        sys.exit(1)
        
    model_name = sys.argv[1]
    result = run_quality_test(model_name)
    
    output_file = f"quality_{model_name.replace('/', '_')}.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)
        
    print(f"\nFinal Score: {result['accuracy_score']}%")
