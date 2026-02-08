import os
import GPUtil
import json

def get_gpu_stats():
    gpus = GPUtil.getGPUs()
    stats = []
    for gpu in gpus:
        stats.append({
            "name": gpu.name,
            "load": gpu.load * 100,
            "memoryTotal": gpu.memoryTotal,
            "memoryUsed": gpu.memoryUsed,
            "temperature": gpu.temperature
        })
    return stats

if __name__ == "__main__":
    stats = get_gpu_stats()
    print(json.dumps(stats, indent=2))
