import numpy as np
import matplotlib.pyplot as plt
import os

# Simulated data for visual feedback loop efficiency
iterations = np.array([1, 2, 3, 4, 5])
hallucination_rate = np.array([42.5, 21.3, 11.2, 5.8, 3.1])
semantic_coherence = np.array([68.2, 75.4, 84.1, 91.5, 96.8])

def generate_charts():
    # Hallucination Rate Chart
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, hallucination_rate, marker='o', linestyle='-', color='red', linewidth=2)
    plt.title('Multimodal Hallucination Rate vs. Feedback Iterations', fontsize=14)
    plt.xlabel('Feedback Iterations (Qwen2-VL Feedback)', fontsize=12)
    plt.ylabel('Hallucination Rate (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('ml-explorations/2026-02-13_self-correcting-multimodal-prompts/charts/hallucination_reduction.png')
    plt.close()

    # Semantic Coherence Chart
    plt.figure(figsize=(10, 6))
    plt.bar(iterations, semantic_coherence, color='teal', alpha=0.8)
    plt.title('Semantic Coherence Gain via Visual Feedback', fontsize=14)
    plt.xlabel('Feedback Iterations', fontsize=12)
    plt.ylabel('Coherence Score (Cosine Similarity)', fontsize=12)
    plt.ylim(60, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig('ml-explorations/2026-02-13_self-correcting-multimodal-prompts/charts/semantic_coherence.png')
    plt.close()

    print("Charts generated successfully.")

if __name__ == "__main__":
    generate_charts()
