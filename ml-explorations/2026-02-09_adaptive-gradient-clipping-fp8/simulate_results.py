import matplotlib.pyplot as plt
import numpy as np

def simulate_training(clipping_type='standard', iterations=100):
    losses = []
    grad_norms = []
    loss = 10.0
    
    for i in range(iterations):
        # Simulate a gradient spike
        if i % 20 == 0 and i > 0:
            grad = np.random.normal(50, 10)
        else:
            grad = np.random.normal(1, 0.2)
            
        if clipping_type == 'adaptive':
            # Adaptive clipping: clip based on running average
            threshold = 2.0 if i < 5 else np.mean(grad_norms[-5:]) * 1.5
            clipped_grad = min(grad, threshold)
        else:
            # Standard clipping
            clipped_grad = min(grad, 5.0)
            
        grad_norms.append(grad)
        loss -= clipped_grad * 0.01 + np.random.normal(0, 0.05)
        losses.append(max(0, loss))
        
    return losses, grad_norms

# Run simulations
std_losses, _ = simulate_training('standard')
adapt_losses, _ = simulate_training('adaptive')

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(std_losses, label='Standard Clipping (5.0)')
plt.plot(adapt_losses, label='Adaptive Clipping (Dynamic)')
plt.title('FP8 Training Stability: Standard vs Adaptive Gradient Clipping')
plt.xlabel('Iterations')
plt.ylabel('Simulated Loss')
plt.legend()
plt.grid(True)
plt.savefig('clipping_comparison.png')
print("Chart generated: clipping_comparison.png")
