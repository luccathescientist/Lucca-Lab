import os
import subprocess

def generate_tests_with_gpt(script_path):
    with open(script_path, 'r') as f:
        code = f.read()
    
    prompt = f"""
You are an expert Python developer. Generate a complete suite of unit tests using the `unittest` framework for the following code.
Return ONLY the Python code for the tests. No explanation.

CODE:
{code}
"""
    # Using openclaw sessions_spawn to simulate GPT-5.2 Codex call via model override
    # But since I am the agent, I will just "think" it or use a sub-agent.
    # For now, I'll use the sessions_spawn to get a high-quality generation.
    return prompt

if __name__ == "__main__":
    script = "target_script.py"
    prompt = generate_tests_with_gpt(script)
    print(prompt)
