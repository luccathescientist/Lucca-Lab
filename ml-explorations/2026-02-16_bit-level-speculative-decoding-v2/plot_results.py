import matplotlib.pyplot as plt

def plot():
    labels = ['Baseline (FP8 Spec)', 'V2 (Bit-Slicing Spec)']
    tps = [58.33, 188.89] # Values from simulation
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, tps, color=['#4285F4', '#34A853'])
    plt.ylabel('Throughput (Tokens Per Second)')
    plt.title('Bit-Level Speculative Decoding Performance on Blackwell (sm_120)')
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.2f} TPS', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('performance_chart.png')
    print("Chart saved as performance_chart.png")

if __name__ == "__main__":
    plot()
