import torch
import time
import subprocess
import json

def get_gpu_stats():
    """Gets GPU memory usage via nvidia-smi."""
    try:
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", 
            "--format=csv,nounits,noheader"
        ]).decode("utf-8")
        used, total, util = map(int, output.strip().split(","))
        return {"used": used, "total": total, "util": util}
    except Exception as e:
        return {"error": str(e)}

def simulate_dynamic_allocation():
    """Simulates loading and unloading models to measure overhead and dynamic thresholds."""
    stats = []
    print("Starting VRAM Allocation Stress Test...")
    
    # Baseline
    stats.append({"event": "baseline", "gpu": get_gpu_stats()})
    
    # Simulate a heavy load (e.g., reserving large tensors)
    try:
        print("Allocating 40GB VRAM...")
        dummy_tensor_1 = torch.cuda.FloatTensor(1024, 1024, 1024 * 10) # ~40GB
        time.sleep(2)
        stats.append({"event": "40gb_allocated", "gpu": get_gpu_stats()})
        
        print("Allocating additional 20GB...")
        dummy_tensor_2 = torch.cuda.FloatTensor(1024, 1024, 1024 * 5) # ~20GB
        time.sleep(2)
        stats.append({"event": "60gb_allocated", "gpu": get_gpu_stats()})
        
        # Simulate 'Flush'
        print("Flushing secondary model memory...")
        del dummy_tensor_2
        torch.cuda.empty_cache()
        time.sleep(2)
        stats.append({"event": "after_flush", "gpu": get_gpu_stats()})
        
    except Exception as e:
        print(f"OOM or Error: {e}")
        stats.append({"event": "error", "error": str(e)})

    with open("vram_benchmark.json", "w") as f:
        json.dump(stats, f, indent=4)
    print("Benchmark complete. Data saved to vram_benchmark.json")

if __name__ == "__main__":
    simulate_dynamic_allocation()
