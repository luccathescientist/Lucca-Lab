import matplotlib.pyplot as plt
import numpy as np
import os

def generate_chart():
    agents = ['R1-70B', 'GPT-5', 'Claude 3.5', 'Consensus (Avg)']
    accuracy = [82, 88, 85, 94]  # Simulated accuracy scores
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(agents, accuracy, color=['#00e5ff', '#ff00ff', '#7000ff', '#ffffff'])
    plt.title('Multi-Agent Consensus Performance (Simulated)', color='white')
    plt.ylabel('Accuracy Score (%)', color='white')
    plt.ylim(0, 100)
    
    # Dark theme styling
    plt.gca().set_facecolor('#0a0a0a')
    plt.gcf().set_facecolor('#0a0a0a')
    plt.tick_params(colors='white')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, yval, ha='center', va='bottom', color='white')

    os.makedirs("ml-explorations/2026-02-09_multi-agent-consensus-distillation/", exist_ok=True)
    plt.savefig('ml-explorations/2026-02-09_multi-agent-consensus-distillation/consensus_chart.png', facecolor='#0a0a0a')
    print("Chart generated: consensus_chart.png")

if __name__ == "__main__":
    generate_chart()
