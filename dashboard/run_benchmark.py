import os
import time
import json
import subprocess
from datetime import datetime

# Path for benchmark results
BENCHMARK_RESULTS_PATH = "/home/user/lab_env/dashboard/benchmark_results.jsonl"

def run_bench():
    """Runs a hardware benchmark and logs the results."""
    print(f"[{datetime.now()}] Starting Hardware Benchmark...")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "gpu": "Unknown",
        "compute": None,
        "memory_bw": None,
        "latency_ms": None,
        "vram_util": None
    }
    
    try:
        # 1. GPU Name and Stats
        cmd = "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits"
        gpu_info = subprocess.check_output(cmd, shell=True).decode('utf-8').strip().split(',')
        results["gpu"] = gpu_info[0].strip()
        vram_total = float(gpu_info[1].strip())
        
        # 2. Performance Metric (Using a simple matrix multiplication via python if available, or just nvidia-smi stats)
        # We'll use nvidia-smi clocks as a proxy for 'compute potential' if we don't want to run heavy cuda kernels
        cmd_clocks = "nvidia-smi --query-gpu=clocks.max.sm,clocks.current.sm --format=csv,noheader,nounits"
        clocks = subprocess.check_output(cmd_clocks, shell=True).decode('utf-8').strip().split(',')
        results["compute"] = f"{clocks[1].strip()}/{clocks[0].strip()} MHz"
        
        # 3. Simulate a Load (Latency Test)
        # We can measure the time it takes to run a small torch operation if torch is in the venv
        start_time = time.time()
        # Simple math loop as a fallback for compute check
        _ = [x*x for x in range(1000000)]
        results["latency_ms"] = round((time.time() - start_time) * 1000, 2)
        
        # 4. Memory Utilization
        cmd_mem = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
        mem_used = float(subprocess.check_output(cmd_mem, shell=True).decode('utf-8').strip())
        results["vram_util"] = f"{round((mem_used / vram_total) * 100, 1)}%"
        
        # Append to results file
        with open(BENCHMARK_RESULTS_PATH, "a") as f:
            f.write(json.dumps(results) + "\n")
            
        print(f"[{datetime.now()}] Benchmark Complete: {results['gpu']} at {results['vram_util']} VRAM utilization.")
        return results

    except Exception as e:
        error_msg = f"Benchmark Error: {str(e)}"
        print(error_msg)
        return {"error": error_msg}

if __name__ == "__main__":
    run_bench()
