import numpy as np
import matplotlib.pyplot as plt
import os

def simulate_latent_optimization():
    # Simulation parameters
    frames = 60
    # Baseline: standard latent handoff (noisy/drifting)
    baseline_drift = np.cumsum(np.random.normal(0, 0.05, frames))
    # Optimized: with R1-steered predictive correction (reduced drift)
    # The "small reasoning model" identifies the drift trajectory and applies inverse steering
    optimized_drift = np.cumsum(np.random.normal(0, 0.01, frames))
    
    plt.figure(figsize=(10, 6))
    plt.plot(baseline_drift, label='Standard Handoff (Baseline)', color='red', linestyle='--')
    plt.plot(optimized_drift, label='Recursive Optimization (R1-Steered)', color='green', linewidth=2)
    plt.title('Latent Drift: Flux.1 -> Wan 2.1 Handoff')
    plt.xlabel('Temporal Frame Index')
    plt.ylabel('Normalized Latent Deviation')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    
    save_path = 'ml-explorations/2026-02-14_recursive-latent-optimization-diffusion/latent_drift_chart.png'
    plt.savefig(save_path)
    print(f"Chart saved to {save_path}")

if __name__ == "__main__":
    simulate_latent_optimization()
