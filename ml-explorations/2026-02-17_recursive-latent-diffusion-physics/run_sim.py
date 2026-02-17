import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def simulate_physics_steering():
    # Simulated physical trajectory (ballistic)
    t = np.linspace(0, 2, 60)
    ideal_y = -4.9 * t**2 + 5 * t + 10  # y = -0.5gt^2 + v0t + y0
    
    # Simulated latent drift (error in video diffusion)
    drift_noise = np.random.normal(0, 0.5, 60).cumsum() * 0.1
    drift_y = ideal_y + drift_noise
    
    # Steering Correction (R1-simulated feedback)
    steering_strength = 0.6
    steered_y = drift_y + steering_strength * (ideal_y - drift_y)
    
    # Calculate MSE
    mse_drift = np.mean((ideal_y - drift_y)**2)
    mse_steered = np.mean((ideal_y - steered_y)**2)
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, ideal_y, 'g--', label='Ideal Physics (Newtonian)')
    plt.plot(t, drift_y, 'r-', label='Unsteered Latent Drift')
    plt.plot(t, steered_y, 'b-', label='R1-Steered Latent (Recursive)')
    plt.title('Recursive Latent-Space Diffusion for Physics-Consistent Video')
    plt.xlabel('Time (s) / Frames')
    plt.ylabel('Vertical Latent Position (y)')
    plt.legend()
    plt.grid(True)
    plt.savefig('ml-explorations/2026-02-17_recursive-latent-diffusion-physics/trajectory_plot.png')
    
    print(f"MSE Drift: {mse_drift:.4f}")
    print(f"MSE Steered: {mse_steered:.4f}")
    print(f"Drift Reduction: {(1 - mse_steered/mse_drift)*100:.2f}%")

if __name__ == "__main__":
    simulate_physics_steering()
