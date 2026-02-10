import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os

def simulate_benchmark(precision, task_complexity):
    """
    Simulates logic benchmark performance based on precision.
    Higher precision (closer to 1.0/FP8) results in higher accuracy.
    Lower precision (closer to 0.5/INT4) results in lower accuracy and higher speed.
    """
    # Base accuracy for the task (simulating R1-32B baseline)
    base_accuracy = 0.88 
    
    if precision == "FP8":
        accuracy = base_accuracy * np.random.uniform(0.98, 1.0)
        latency = np.random.uniform(20, 30) # ms/token
    elif precision == "INT8":
        accuracy = base_accuracy * np.random.uniform(0.94, 0.97)
        latency = np.random.uniform(15, 22)
    elif precision == "INT4":
        accuracy = base_accuracy * np.random.uniform(0.80, 0.88)
        latency = np.random.uniform(8, 14)
    else:
        accuracy = 0.5
        latency = 100

    # Adjust for complexity
    accuracy -= (task_complexity * 0.05)
    
    return accuracy, latency

def run_suite():
    precisions = ["FP8", "INT8", "INT4"]
    complexities = [1, 2, 3, 4, 5] # 1: Basic Logic, 5: Multi-step Calculus
    
    results = {}
    
    for p in precisions:
        results[p] = {"accuracy": [], "latency": []}
        for c in complexities:
            acc, lat = simulate_benchmark(p, c)
            results[p]["accuracy"].append(acc)
            results[p]["latency"].append(lat)
            
    # Save raw data
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Plotting Accuracy
    plt.figure(figsize=(10, 6))
    for p in precisions:
        plt.plot(complexities, results[p]["accuracy"], marker='o', label=f'{p} Accuracy')
    
    plt.title("IQ Loss vs. Precision (Simulated R1-32B)")
    plt.xlabel("Task Complexity (1-5)")
    plt.ylabel("Accuracy Score")
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_chart.png")
    
    # Plotting Latency
    plt.figure(figsize=(10, 6))
    for p in precisions:
        plt.bar(p, np.mean(results[p]["latency"]), label=f'{p} Avg Latency')
    
    plt.title("Inference Latency vs. Precision")
    plt.ylabel("Latency (ms/token)")
    plt.savefig("latency_chart.png")

if __name__ == "__main__":
    print("Starting Quantized-Logic Reasoning Benchmarks...")
    run_suite()
    print("Benchmark complete. Data and charts saved.")
