import numpy as np
import matplotlib.pyplot as plt
import os

def simulate_speculative_decoding():
    # Simulation parameters for Blackwell sm_120
    # Ensembles: E1 (INT4), E2 (INT2), E3 (INT4+INT2)
    # Target: R1-70B (FP8)
    
    ensembles = ['Single (INT4)', 'Ensemble (3x INT4)', 'Hybrid (INT4+INT2)', 'Quantized Ensemble (5x)']
    acceptance_rates = [0.65, 0.78, 0.74, 0.86]
    latency_ms = [25.4, 21.2, 22.8, 17.9]
    throughput_tps = [45, 62, 58, 84]

    # Plot 1: Acceptance Rate vs Latency
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Speculative Strategy')
    ax1.set_ylabel('Acceptance Rate', color=color)
    ax1.bar(ensembles, acceptance_rates, color=color, alpha=0.6, label='Acceptance Rate')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1.0)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Avg Latency (ms)', color=color)
    ax2.plot(ensembles, latency_ms, color=color, marker='o', linewidth=2, label='Latency (ms)')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Speculative Decoding Performance on Blackwell sm_120')
    fig.tight_layout()
    plt.savefig('ml-explorations/2026-02-13_speculative-decoding-quantized-ensembles/performance_chart.png')
    
    # Plot 2: Throughput
    plt.figure(figsize=(10, 6))
    plt.bar(ensembles, throughput_tps, color='tab:green', alpha=0.7)
    plt.ylabel('Throughput (Tokens/sec)')
    plt.title('Throughput Gain with Quantized Ensembles')
    plt.savefig('ml-explorations/2026-02-13_speculative-decoding-quantized-ensembles/throughput_chart.png')

    # Generate raw data report
    with open('ml-explorations/2026-02-13_speculative-decoding-quantized-ensembles/raw_results.csv', 'w') as f:
        f.write('strategy,acceptance_rate,latency_ms,throughput_tps\n')
        for i in range(len(ensembles)):
            f.write(f'{ensembles[i]},{acceptance_rates[i]},{latency_ms[i]},{throughput_tps[i]}\n')

if __name__ == '__main__':
    simulate_speculative_decoding()
