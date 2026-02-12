import numpy as np
import matplotlib.pyplot as plt
import os

# Simulated data for Recursive Self-Correction (RSC)
# We compare a standard Multimodal Reasoning pass vs RSC (3-iteration loop)

iterations = np.array([0, 1, 2, 3])
hallucination_rate = np.array([0.45, 0.28, 0.12, 0.04])  # Simulated decline in logical errors
confidence_score = np.array([0.62, 0.78, 0.89, 0.96])   # Simulated increase in grounding confidence
latency_ms = np.array([45, 112, 178, 245])             # Cumulative latency on sm_120

def generate_charts(output_dir):
    # Plot 1: Hallucination Rate vs Iterations
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, hallucination_rate * 100, marker='o', linestyle='-', color='red', linewidth=2, label='Hallucination Rate (%)')
    plt.title('Recursive Self-Correction: Hallucination Mitigation (Simulated)', fontsize=14)
    plt.xlabel('Correction Iterations', fontsize=12)
    plt.ylabel('Rate (%)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'hallucination_reduction.png'), dpi=300)
    plt.close()

    # Plot 2: Confidence vs Latency Tradeoff
    plt.figure(figsize=(10, 6))
    plt.scatter(latency_ms, confidence_score, s=100, c='blue', alpha=0.7)
    for i, txt in enumerate(iterations):
        plt.annotate(f"Iter {txt}", (latency_ms[i], confidence_score[i]), textcoords="offset points", xytext=(0,10), ha='center')
    plt.plot(latency_ms, confidence_score, linestyle='--', color='blue', alpha=0.3)
    plt.title('Grounding Confidence vs. Inference Latency (Blackwell sm_120)', fontsize=14)
    plt.xlabel('Cumulative Latency (ms)', fontsize=12)
    plt.ylabel('Grounding Confidence Score', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, 'confidence_latency_tradeoff.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    output_path = "ml-explorations/2026-02-13_recursive-self-correction-multimodal-hallucinations"
    generate_charts(output_path)
    print(f"Charts generated in {output_path}")
