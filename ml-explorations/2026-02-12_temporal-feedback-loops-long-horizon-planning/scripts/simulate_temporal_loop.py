import numpy as np
import matplotlib.pyplot as plt
import time

def simulate_temporal_feedback_loop():
    # Simulation parameters
    total_steps = 100
    base_accuracy = 0.65
    memory_buffer_decay = 0.05
    feedback_strength = 0.15
    
    # State tracking
    reasoning_accuracy = []
    consistency_score = []
    memory_utility = []
    
    current_acc = base_accuracy
    memory_influence = 0.0
    
    for step in range(total_steps):
        # Temporal buffer update (simulated)
        # Utility peaks as context builds, then stabilizes
        memory_influence = np.tanh(step / 20.0) * (1.0 - memory_buffer_decay)
        
        # Accuracy gains from feedback loop
        noise = np.random.normal(0, 0.02)
        current_acc = base_accuracy + (feedback_strength * memory_influence) + noise
        
        # Consistency score (simulated as reduction in variance)
        consistency = 1.0 - (0.35 * np.exp(-step / 40.0))
        
        reasoning_accuracy.append(current_acc)
        consistency_score.append(consistency)
        memory_utility.append(memory_influence)
        
    return reasoning_accuracy, consistency_score, memory_utility

def plot_results(acc, cons, util):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Reasoning Accuracy', color='blue')
    plt.axhline(y=0.65, color='red', linestyle='--', label='Baseline')
    plt.title('Reasoning Accuracy with Temporal Feedback')
    plt.xlabel('Reasoning Steps')
    plt.ylabel('Accuracy Score')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(cons, label='Consistency Score', color='green')
    plt.plot(util, label='Memory Utility', color='purple', linestyle=':')
    plt.title('Consistency and Memory Utility')
    plt.xlabel('Reasoning Steps')
    plt.ylabel('Normalized Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('ml-explorations/2026-02-12_temporal-feedback-loops-long-horizon-planning/temporal_reasoning_performance.png')
    print("Chart generated: temporal_reasoning_performance.png")

if __name__ == "__main__":
    print("Starting Temporal Feedback Loop Simulation...")
    acc, cons, util = simulate_temporal_feedback_loop()
    plot_results(acc, cons, util)
    print("Simulation Complete.")
