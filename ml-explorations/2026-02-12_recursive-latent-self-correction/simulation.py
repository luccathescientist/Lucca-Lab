import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def run_simulation(dim=128, iters=10):
    # Simulate a high-dimensional latent vector
    # Represents the 'reasoning state'
    latent = np.random.randn(dim)
    
    # Target 'consistent' state (simulated)
    target = np.random.randn(dim) * 0.1 
    
    latents_history = [latent.copy()]
    
    # Simulate Blackwell Tensor Core optimization:
    # We use a simulated 'gradient' of logical consistency
    for i in range(iters):
        # Calculate consistency error
        error = target - latent
        
        # Simulated recursive update: 
        # In a real model, this would be a second 'verifier' head
        # outputting a correction vector in latent space.
        correction_strength = 0.2
        noise_level = 0.05 / (i + 1) # Error decreases over iterations
        
        latent += correction_strength * error + np.random.randn(dim) * noise_level
        latents_history.append(latent.copy())
        
    # Calculate "Logical Inconsistency" as MSE to target
    mse_history = [np.mean((l - target)**2) for l in latents_history]
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(mse_history)), mse_history, marker='o', linestyle='-', color='#00FFCC')
    plt.yscale('log')
    plt.title('Recursive Latent Self-Correction Convergence', fontsize=14, color='white')
    plt.xlabel('Correction Iterations', fontsize=12, color='white')
    plt.ylabel('Logical Inconsistency (MSE)', fontsize=12, color='white')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Style the plot for a dark theme lab look
    plt.gcf().set_facecolor('#1E1E1E')
    plt.gca().set_facecolor('#2D2D2D')
    plt.tick_params(colors='white')
    
    plt.savefig('convergence_plot.png')
    
    return mse_history

if __name__ == "__main__":
    history = run_simulation()
    print(f"Initial MSE: {history[0]:.6f}")
    print(f"Final MSE: {history[-1]:.6f}")
    print(f"Convergence Improvement: {((history[0] - history[-1]) / history[0]) * 100:.2f}%")
