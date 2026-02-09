import matplotlib.pyplot as plt
import numpy as np
import os

def generate_chart():
    np.random.seed(42)
    tokens = np.arange(128)
    entropy = np.abs(np.random.randn(128) * 2 + 3)  # Simulated entropy values
    
    # Flag high uncertainty tokens (above 95th percentile)
    threshold = np.percentile(entropy, 95)
    high_uncertainty = entropy > threshold
    
    plt.figure(figsize=(14, 5))
    plt.bar(tokens[~high_uncertainty], entropy[~high_uncertainty], color='#00e5ff', label='Normal Tokens')
    plt.bar(tokens[high_uncertainty], entropy[high_uncertainty], color='#ff3860', label='High Uncertainty (Potential Hallucination)')
    
    plt.axhline(y=threshold, color='#ffdd57', linestyle='--', label=f'95th Percentile Threshold ({threshold:.2f})')
    
    plt.title('Token-Level Uncertainty Quantification (Simulated)', color='white')
    plt.xlabel('Token Index', color='white')
    plt.ylabel('Entropy (Uncertainty)', color='white')
    plt.legend(facecolor='#1a1a2e')
    
    # Dark theme
    plt.gca().set_facecolor('#0a0a0a')
    plt.gcf().set_facecolor('#0a0a0a')
    plt.tick_params(colors='white')
    plt.legend(facecolor='#1a1a2e', labelcolor='white')
    
    os.makedirs("ml-explorations/2026-02-09_token-level-uncertainty-quantification/", exist_ok=True)
    plt.savefig('ml-explorations/2026-02-09_token-level-uncertainty-quantification/uncertainty_chart.png', facecolor='#0a0a0a', bbox_inches='tight')
    print("Chart generated: uncertainty_chart.png")

if __name__ == "__main__":
    generate_chart()
