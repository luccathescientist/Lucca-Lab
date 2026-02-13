import matplotlib.pyplot as plt
import numpy as np

def generate_performance_chart():
    categories = ['Base R1', 'Speculative (S1)', 'Speculative (S1+S2)', 'Recursive KG Boost']
    latency = [120, 45, 32, 28] # ms
    accuracy = [98, 97.8, 97.7, 99.2] # %

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Latency (ms)', color=color)
    ax1.bar(categories, latency, color=color, alpha=0.6, label='Latency')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Accuracy (%)', color=color)
    ax2.plot(categories, accuracy, color=color, marker='o', linewidth=2, label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(95, 100)

    plt.title('Performance Gains from Recursive KG Expansion on Blackwell sm_120')
    fig.tight_layout()
    plt.savefig('performance_gains.png')
    print("Chart saved as performance_gains.png")

if __name__ == "__main__":
    generate_performance_chart()
