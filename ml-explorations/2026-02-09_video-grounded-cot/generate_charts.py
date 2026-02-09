import matplotlib.pyplot as plt
import numpy as np

def generate_performance_chart():
    # Simulated data: Hallucination rate (%) and Latency (ms)
    # Baseline: R1 (Vanilla)
    # Grounded: R1 + Video Grounding
    
    categories = ['Vanilla R1', 'Video-Grounded R1']
    hallucination_rate = [22.5, 4.2] # Lower is better
    latency = [120, 450] # Higher is worse (due to vision overhead)

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Model Strategy')
    ax1.set_ylabel('Hallucination Rate (%)', color=color)
    ax1.bar(categories, hallucination_rate, color=color, alpha=0.6, label='Hallucination %')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Inference Latency (ms)', color=color)
    ax2.plot(categories, latency, color=color, marker='o', label='Latency (ms)')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Video-Grounded CoT: Accuracy vs. Overhead')
    fig.tight_layout()
    plt.savefig('ml-explorations/2026-02-09_video-grounded-cot/performance_chart.png')
    print("Performance chart generated.")

if __name__ == "__main__":
    generate_performance_chart()
