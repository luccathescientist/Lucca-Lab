
import requests, json, time, os, random

COMFY_URL = 'http://127.0.0.1:8188'
PROMPT_FILE = '/workspace/workflow_flux_simple.json'
ITERATIONS = 10

def benchmark():
    with open(PROMPT_FILE, 'r') as f:
        workflow = json.load(f)

    times = []
    print(f"Starting benchmark for {ITERATIONS} images...")
    
    # Ensure ComfyUI is warmed up (it already is from previous run, but good practice)
    
    for i in range(ITERATIONS):
        start_time = time.time()
        
        # Modify seed and prompt slightly
        workflow['3']['inputs']['seed'] = random.randint(0, 1000000)
        workflow['6']['inputs']['text'] = f"A small sci-fi gadget, glowing crystal core, highly detailed, photorealistic, iteration {i+1}"
        
        response = requests.post(f'{COMFY_URL}/prompt', json={'prompt': workflow})
        prompt_id = response.json()['prompt_id']
        
        while True:
            history = requests.get(f'{COMFY_URL}/history/{prompt_id}').json()
            if prompt_id in history:
                end_time = time.time()
                duration = end_time - start_time
                times.append(duration)
                print(f"Image {i+1}/{ITERATIONS}: {duration:.2f}s")
                break
            time.sleep(0.5)

    avg_time = sum(times) / len(times)
    total_time = sum(times)
    print(f"\nBenchmark Complete!")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average Time per Image: {avg_time:.2f}s")
    print(f"Min Time: {min(times):.2f}s")
    print(f"Max Time: {max(times):.2f}s")
    
    # Return the last image path for confirmation
    output = history[prompt_id]['outputs']
    node_id = list(output.keys())[0]
    filename = output[node_id]['images'][0]['filename']
    print(f"LAST_IMAGE_PATH:/workspace/ComfyUI/output/{filename}")

if __name__ == "__main__":
    benchmark()
