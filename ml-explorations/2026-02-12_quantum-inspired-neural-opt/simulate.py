import numpy as np
import matplotlib.pyplot as plt
import time

def objective_function(x):
    """A complex non-convex landscape for optimization testing."""
    return (x**2 - 10 * np.cos(2 * np.pi * x)) + ( (x-2)**2 - 10 * np.cos(2 * np.pi * (x-2)) )

def simulated_annealing(objective, bounds, n_iterations, step_size, temp):
    best = np.random.uniform(bounds[0], bounds[1])
    best_eval = objective(best)
    curr, curr_eval = best, best_eval
    
    history = [(curr, curr_eval)]
    
    for i in range(n_iterations):
        # Metropolis step
        candidate = curr + np.random.normal(0, step_size)
        candidate = np.clip(candidate, bounds[0], bounds[1])
        candidate_eval = objective(candidate)
        
        if candidate_eval < best_eval:
            best, best_eval = candidate, candidate_eval
            
        diff = candidate_eval - curr_eval
        t = temp / float(i + 1)
        metropolis = np.exp(-diff / t)
        
        if diff < 0 or np.random.rand() < metropolis:
            curr, curr_eval = candidate, candidate_eval
            
        history.append((curr, curr_eval))
        
    return best, best_eval, history

def quantum_inspired_annealing(objective, bounds, n_iterations, step_size, temp, n_particles=5):
    """Simulates Quantum Tunneling by maintaining multiple 'parallel' paths that can influence each other."""
    particles = np.random.uniform(bounds[0], bounds[1], n_particles)
    particle_evals = np.array([objective(p) for p in particles])
    
    best_idx = np.argmin(particle_evals)
    best, best_eval = particles[best_idx], particle_evals[best_idx]
    
    history = [(best, best_eval)]
    
    for i in range(n_iterations):
        t = temp / float(i + 1)
        
        for j in range(n_particles):
            # Quantum-like step: move toward best but with high variance (tunneling)
            tunneling_factor = np.random.normal(0, step_size * (1 + t))
            candidate = particles[j] + tunneling_factor
            
            # Influence from 'entangled' global best
            entanglement = (best - particles[j]) * np.random.rand() * 0.1
            candidate += entanglement
            
            candidate = np.clip(candidate, bounds[0], bounds[1])
            candidate_eval = objective(candidate)
            
            if candidate_eval < particle_evals[j] or np.random.rand() < np.exp(-(candidate_eval - particle_evals[j]) / t):
                particles[j], particle_evals[j] = candidate, candidate_eval
                
            if candidate_eval < best_eval:
                best, best_eval = candidate, candidate_eval
        
        history.append((best, best_eval))
        
    return best, best_eval, history

# Simulation params
bounds = [-10, 10]
n_iter = 1000
step = 0.5
temp = 10.0

# Run SA
sa_best, sa_val, sa_history = simulated_annealing(objective_function, bounds, n_iter, step, temp)

# Run QIA
qia_best, qia_val, qia_history = quantum_inspired_annealing(objective_function, bounds, n_iter, step, temp)

# Plotting
x = np.linspace(bounds[0], bounds[1], 1000)
y = objective_function(x)

plt.figure(figsize=(12, 6))
plt.plot(x, y, label='Objective Landscape', alpha=0.5)
plt.scatter([h[0] for h in sa_history[::10]], [h[1] for h in sa_history[::10]], color='red', s=10, label='SA Path (Every 10th)')
plt.scatter([h[0] for h in qia_history[::10]], [h[1] for h in qia_history[::10]], color='blue', s=10, label='QIA Path (Every 10th)')
plt.title("Quantum-Inspired vs Simulated Annealing")
plt.legend()
plt.savefig('ml-explorations/2026-02-12_quantum-inspired-neural-opt/plots/comparison.png')

# Convergence Plot
plt.figure(figsize=(10, 5))
plt.plot([h[1] for h in sa_history], label='SA Convergence', color='red')
plt.plot([h[1] for h in qia_history], label='QIA Convergence', color='blue')
plt.yscale('log')
plt.title("Convergence Rate (Log Scale)")
plt.legend()
plt.savefig('ml-explorations/2026-02-12_quantum-inspired-neural-opt/plots/convergence.png')

print(f"SA Best: {sa_val:.4f} at {sa_best:.4f}")
print(f"QIA Best: {qia_val:.4f} at {qia_best:.4f}")
