import numpy as np
import matplotlib.pyplot as plt

def simulate_dpo_stability(precision_bits, beta_values):
    # Simulated stability metric (higher is better)
    # Model: Stability = f(precision, beta)
    results = {}
    for bits in precision_bits:
        stability = []
        for beta in beta_values:
            # Theoretical relationship: lower bits require higher beta to regularize quantization noise
            # but too high beta kills learning.
            noise_factor = 1.0 / (2**bits)
            score = 1.0 - (noise_factor / (beta + 0.05)) - (0.01 * beta**2)
            stability.append(max(0, score))
        results[bits] = stability
    return results

def plot_results(results, beta_values):
    plt.figure(figsize=(10, 6))
    for bits, stability in results.items():
        plt.plot(beta_values, stability, label=f'{bits}-bit Precision', marker='o')
    
    plt.title('Hardware-Aware DPO Stability vs. Beta')
    plt.xlabel('Beta (Regularization Strength)')
    plt.ylabel('Training Stability Score')
    plt.legend()
    plt.grid(True)
    plt.savefig('ml-explorations/2026-02-12_hardware-aware-dpo-sub-int4/dpo_stability.png')
    plt.close()

if __name__ == "__main__":
    precision_bits = [2, 3, 4, 8]
    beta_values = np.linspace(0.01, 0.5, 10)
    results = simulate_dpo_stability(precision_bits, beta_values)
    plot_results(results, beta_values)
    print("Simulation complete. Chart generated.")
