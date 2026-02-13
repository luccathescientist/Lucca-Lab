import numpy as np
import matplotlib.pyplot as plt
import os

def simulate_drift(method="baseline", steps=100):
    np.random.seed(42)
    drift = np.zeros(steps)
    current_drift = 0
    
    if method == "baseline":
        # Random walk drift
        for i in range(steps):
            current_drift += np.random.normal(0, 0.05)
            drift[i] = current_drift
    elif method == "quantum_inspired":
        # Simulated annealing/quantum tunneling approach
        # Drift is corrected towards a global optimum (zero drift)
        temperature = 1.0
        for i in range(steps):
            candidate = current_drift + np.random.normal(0, 0.05)
            # Energy function is the drift itself
            if abs(candidate) < abs(current_drift):
                current_drift = candidate
            else:
                # Probability of accepting worse drift decreases over time (annealing)
                if np.random.rand() < np.exp(-(abs(candidate) - abs(current_drift)) / temperature):
                    current_drift = candidate
            drift[i] = current_drift
            temperature *= 0.95 # Cooling
            
    return drift

def run_experiment():
    steps = 50
    baseline = simulate_drift("baseline", steps)
    quantum = simulate_drift("quantum_inspired", steps)
    
    plt.figure(figsize=(10, 6))
    plt.plot(baseline, label='Baseline Latent Handoff', color='red', linestyle='--')
    plt.plot(quantum, label='Quantum-Inspired Diffusion (Annealing)', color='cyan', linewidth=2)
    plt.title('Latent Identity Drift: Baseline vs Quantum-Inspired Diffusion')
    plt.xlabel('Handoff Refinement Steps')
    plt.ylabel('Semantic Drift (L2 Norm)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    output_path = 'ml-explorations/2026-02-13_quantum-inspired-diffusion-latent-handoffs/drift_comparison.png'
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")

    # Calculate metrics
    final_baseline_drift = abs(baseline[-1])
    final_quantum_drift = abs(quantum[-1])
    improvement = (1 - (final_quantum_drift / final_baseline_drift)) * 100
    
    with open('ml-explorations/2026-02-13_quantum-inspired-diffusion-latent-handoffs/metrics.txt', 'w') as f:
        f.write(f"Final Baseline Drift: {final_baseline_drift:.4f}\n")
        f.write(f"Final Quantum Drift: {final_quantum_drift:.4f}\n")
        f.write(f"Improvement: {improvement:.2f}%\n")

if __name__ == "__main__":
    run_experiment()
