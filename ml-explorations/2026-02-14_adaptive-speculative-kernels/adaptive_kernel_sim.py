import matplotlib.pyplot as plt
import numpy as np

def simulate_throughput(fp8_ratio):
    # Performance constants for RTX 6000 Blackwell (theoretical)
    FP8_TFLOPS = 1800  # FP8 Tensor Core throughput
    INT4_TFLOPS = 3600 # INT4 Tensor Core throughput (theoretical 2x vs FP8)
    
    # Adaptive kernel overhead (swapping/dispatch)
    overhead_ms = 0.05 
    
    # Calculate effective throughput based on mix
    # Blackwell can execute different precisions on different TPCs or 
    # use specialized paths for dual-precision passes.
    throughput = (fp8_ratio * FP8_TFLOPS) + ((1 - fp8_ratio) * INT4_TFLOPS)
    
    # Penalize for switching frequency (mocking real-world overhead)
    throughput *= (1 - (overhead_ms / 1.0)) 
    
    return throughput

def run_experiment():
    ratios = np.linspace(0, 1, 11)
    throughputs = [simulate_throughput(r) for r in ratios]
    
    plt.figure(figsize=(10, 6))
    plt.plot(ratios * 100, throughputs, marker='o', linestyle='-', color='teal', label='Adaptive Throughput')
    plt.axhline(y=1800, color='red', linestyle='--', label='Baseline FP8')
    plt.title('Blackwell Adaptive Speculative Kernel Throughput (FP8/INT4 Mix)')
    plt.xlabel('FP8 Tensor Ratio (%)')
    plt.ylabel('Effective TFLOPS')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('throughput_chart.png')
    
    # Output raw data for the report
    with open('raw_data.txt', 'w') as f:
        f.write("FP8_Ratio,Effective_TFLOPS\n")
        for r, t in zip(ratios, throughputs):
            f.write(f"{r:.2f},{t:.2f}\n")

if __name__ == "__main__":
    run_experiment()
