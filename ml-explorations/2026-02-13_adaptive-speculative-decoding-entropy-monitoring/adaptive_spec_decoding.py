import numpy as np
import time
import matplotlib.pyplot as plt

def simulate_entropy_monitoring():
    """
    Simulates real-time entropy monitoring and model switching for speculative decoding.
    Target Model: R1-70B (Deep Reasoning)
    Students: 
      - Student A: R1-1.5B (Fast, low entropy tasks)
      - Student B: R1-7B (Medium, moderate complexity)
      - Student C: R1-14B (Robust, high entropy / complex logic)
    """
    
    # Simulation Parameters
    num_steps = 100
    entropy_threshold_low = 0.4
    entropy_threshold_high = 0.8
    
    # Simulated Entropies (representing varying complexity of tokens)
    # High entropy = high uncertainty/complexity
    simulated_entropies = np.abs(np.sin(np.linspace(0, 4 * np.pi, num_steps)) * 0.5 + np.random.normal(0.3, 0.1, num_steps))
    
    selected_students = []
    latencies = []
    acceptance_rates = []
    
    for entropy in simulated_entropies:
        if entropy < entropy_threshold_low:
            # Student A (Smallest/Fastest)
            student = "R1-1.5B"
            latency = 10.5 # ms
            acc_rate = 0.85 - (entropy * 0.2)
        elif entropy < entropy_threshold_high:
            # Student B (Medium)
            student = "R1-7B"
            latency = 25.2 # ms
            acc_rate = 0.90 - (entropy * 0.1)
        else:
            # Student C (Large/Slowest but most accurate)
            student = "R1-14B"
            latency = 45.8 # ms
            acc_rate = 0.95 - (entropy * 0.05)
            
        selected_students.append(student)
        latencies.append(latency)
        acceptance_rates.append(acc_rate)
        
    return simulated_entropies, selected_students, latencies, acceptance_rates

def generate_report_visuals(entropies, students, latencies, acceptance_rates):
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Entropy and Student Selection
    plt.subplot(2, 1, 1)
    colors = {'R1-1.5B': 'green', 'R1-7B': 'orange', 'R1-14B': 'red'}
    for i in range(len(entropies)):
        plt.scatter(i, entropies[i], color=colors[students[i]], alpha=0.6)
    
    plt.axhline(y=0.4, color='gray', linestyle='--', label='Low Entropy Threshold')
    plt.axhline(y=0.8, color='black', linestyle='--', label='High Entropy Threshold')
    plt.title("Adaptive Student Selection based on Token Entropy")
    plt.ylabel("Entropy")
    plt.legend(["Low Threshold", "High Threshold", "1.5B (Fast)", "7B (Mid)", "14B (Robust)"])
    
    # Plot 2: Acceptance Rate vs Latency (Normalized)
    plt.subplot(2, 1, 2)
    plt.plot(latencies, label="Latency (ms)", color='blue')
    plt.plot(np.array(acceptance_rates) * 50, label="Acceptance Rate (Scaled x50)", color='purple')
    plt.title("Latency vs Acceptance Rate Trade-off")
    plt.xlabel("Decoding Step")
    plt.ylabel("Metric Value")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("ml-explorations/2026-02-13_adaptive-speculative-decoding-entropy-monitoring/performance_metrics.png")
    print("Visuals generated at: ml-explorations/2026-02-13_adaptive-speculative-decoding-entropy-monitoring/performance_metrics.png")

if __name__ == "__main__":
    e, s, l, a = simulate_entropy_monitoring()
    generate_report_visuals(e, s, l, a)
    
    avg_latency = np.mean(l)
    avg_acc = np.mean(a)
    print(f"Simulation Complete.")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Average Acceptance Rate: {avg_acc:.2%}")
