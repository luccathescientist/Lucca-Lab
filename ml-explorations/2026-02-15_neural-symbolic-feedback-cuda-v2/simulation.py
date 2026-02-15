import numpy as np
import matplotlib.pyplot as plt
import os

# Simulated Data for R1-driven Neural Symbolic Feedback (v2) on Blackwell sm_120
iterations = np.arange(1, 11)
memory_safety_violations = [12, 8, 4, 1, 0, 0, 0, 0, 0, 0]
race_conditions = [15, 9, 5, 2, 0, 0, 0, 0, 0, 0]
l2_cache_utilization = [0.65, 0.72, 0.78, 0.85, 0.91, 0.92, 0.92, 0.93, 0.92, 0.92]
pflops_performance = [1.1, 1.25, 1.38, 1.45, 1.58, 1.62, 1.64, 1.65, 1.65, 1.65]

def plot_verification_progress():
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Feedback Loop Iteration')
    ax1.set_ylabel('Errors Detected (Z3 Symbolic Solver)', color=color)
    ax1.plot(iterations, memory_safety_violations, label='OOB Errors', color='tab:red', marker='o', linestyle='--')
    ax1.plot(iterations, race_conditions, label='Race Conditions', color='tab:orange', marker='s', linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Performance / Utilization', color=color)
    ax2.plot(iterations, l2_cache_utilization, label='L2 Cache Utilization', color='tab:blue', marker='^')
    ax2.plot(iterations, [p/2.0 for p in pflops_performance], label='Normalized PFLOPS (x0.5)', color='tab:cyan', marker='v')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Neural Symbolic Feedback (v2): Zero-Error Convergence for Blackwell sm_120')
    fig.tight_layout()
    plt.legend(loc='upper right')
    
    save_path = 'ml-explorations/2026-02-15_neural-symbolic-feedback-cuda-v2/performance_chart.png'
    plt.savefig(save_path)
    print(f"Chart saved to {save_path}")

if __name__ == "__main__":
    plot_verification_progress()
