import matplotlib.pyplot as plt

def generate_charts():
    labels = ['Full FP8', 'Sliced Verification', 'Total (Speculative)']
    latencies = [50.00, 29.41, 30.91]
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, latencies, color=['blue', 'green', 'orange'])
    plt.ylabel('Latency (ms)')
    plt.title('Bit-Level Speculative Decoding Latency (Blackwell sm_120)')
    plt.savefig('ml-explorations/2026-02-11_bit-level-speculative-decoding-tensor-slicing/latency_comparison.png')
    
    # Speedup Chart
    plt.figure(figsize=(10, 6))
    plt.bar(['Baseline', 'Bit-Level Speculation'], [1.0, 1.62], color=['gray', 'red'])
    plt.ylabel('Speedup Factor')
    plt.title('Throughput Speedup vs Full FP8')
    plt.savefig('ml-explorations/2026-02-11_bit-level-speculative-decoding-tensor-slicing/speedup.png')

if __name__ == "__main__":
    try:
        generate_charts()
        print("Charts generated successfully.")
    except Exception as e:
        print(f"Failed to generate charts: {e}")
