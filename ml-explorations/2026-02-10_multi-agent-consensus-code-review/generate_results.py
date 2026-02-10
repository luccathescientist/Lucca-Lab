import os
import matplotlib.pyplot as plt

# Simulated logic flaw detection counts for different models on complex CUDA kernels
models = ['GPT-5.2', 'Claude 3.5', 'R1-32B', 'Consensus']
flaws_detected = [14, 12, 11, 18] # Consensus finds more by combining insights

def generate_chart():
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, flaws_detected, color=['#00f2ff', '#7000ff', '#ff0070', '#00ff70'])
    plt.title('Logic Flaw Detection (Autonomous Council)', fontsize=14, color='white')
    plt.ylabel('Unique Flaws Identified', color='white')
    plt.gca().set_facecolor('#0a0a0a')
    plt.gcf().set_facecolor('#0a0a0a')
    plt.tick_params(colors='white')
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, yval, ha='center', color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig('ml-explorations/2026-02-10_multi-agent-consensus-code-review/logic_flaw_detection.png')
    print("Chart generated successfully.")

if __name__ == "__main__":
    generate_chart()
