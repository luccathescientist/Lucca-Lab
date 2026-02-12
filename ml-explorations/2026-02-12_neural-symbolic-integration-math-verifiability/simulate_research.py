import json
import re
import numpy as np
import matplotlib.pyplot as plt

def simulate_logic_verification():
    """
    Simulates the Neural Symbolic Integration pipeline.
    R1-driven reasoning vs. Symbolic Solver verification.
    """
    # Parameters for the simulation
    steps = 100
    r1_accuracy_base = 0.78  # Baseline for R1 on complex math
    symbolic_verifiability = 0.95 # Accuracy of the solver
    
    # Tracking performance over 'iterations' of DPO feedback
    iterations = 5
    results = []
    
    for i in range(iterations):
        # As DPO penalties are applied, the model's 'logical' alignment increases
        r1_acc = r1_accuracy_base + (0.18 * (1 - np.exp(-i/2)))
        hallucination_rate = 0.25 * np.exp(-i/1.5)
        
        # Simulated verification pass
        detected_errors = hallucination_rate * symbolic_verifiability
        corrected_acc = r1_acc + (detected_errors * 0.8) # Partial recovery via feedback
        
        results.append({
            "iteration": i,
            "r1_raw_acc": r1_acc,
            "hallucination_rate": hallucination_rate,
            "corrected_acc": corrected_acc
        })
        
    return results

def plot_results(results):
    iters = [r['iteration'] for r in results]
    raw_acc = [r['r1_raw_acc'] for r in results]
    hall_rate = [r['hallucination_rate'] for r in results]
    corr_acc = [r['corrected_acc'] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(iters, raw_acc, label='R1 Raw Accuracy', marker='o', linestyle='--')
    plt.plot(iters, corr_acc, label='Integrated (Neural+Symbolic) Accuracy', marker='s')
    plt.plot(iters, hall_rate, label='Hallucination Rate', marker='x', color='red')
    
    plt.title('Neural Symbolic Integration: Math Verifiability Performance')
    plt.xlabel('DPO Feedback Iteration')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig('ml-explorations/2026-02-12_neural-symbolic-integration-math-verifiability/performance_chart.png')
    plt.close()

if __name__ == "__main__":
    print("Starting Neural Symbolic Integration simulation...")
    data = simulate_logic_verification()
    plot_results(data)
    
    with open('ml-explorations/2026-02-12_neural-symbolic-integration-math-verifiability/data.json', 'w') as f:
        json.dump(data, f, indent=4)
    
    print("Simulation complete. Data and chart saved.")
