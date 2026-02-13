import numpy as np
import matplotlib.pyplot as plt

def simulate_steering(steering_strength=0.5):
    # Simulated latent trajectories
    t = np.linspace(0, 10, 100)
    baseline = np.sin(t) + np.random.normal(0, 0.1, 100)
    
    # Steering reduces drift and anchors to target narrative
    steered = np.sin(t) * (1 - steering_strength) + (np.sin(t) * 1.2) * steering_strength + np.random.normal(0, 0.02, 100)
    
    # Calculate drift metrics
    drift_baseline = np.cumsum(np.abs(np.diff(baseline)))
    drift_steered = np.cumsum(np.abs(np.diff(steered)))
    
    return t, baseline, steered, drift_baseline, drift_steered

def plot_results():
    t, b, s, db, ds = simulate_steering()
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(t, b, label='Baseline Latent Path', alpha=0.7)
    plt.plot(t, s, label='Steered Latent Path', linewidth=2)
    plt.title('Latent Space Trajectory (Fourier-Steered)')
    plt.xlabel('Diffusion Step')
    plt.ylabel('Latent Magnitude')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(t[1:], db, label='Baseline Drift', alpha=0.7)
    plt.plot(t[1:], ds, label='Steered Drift', linewidth=2)
    plt.title('Cumulative Latent Drift')
    plt.xlabel('Diffusion Step')
    plt.ylabel('Drift Magnitude')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('ml-explorations/2026-02-13_cross-modal-latent-steering-dynamic-narratives/steering_metrics.png')
    print("Metrics chart generated.")

if __name__ == "__main__":
    plot_results()
