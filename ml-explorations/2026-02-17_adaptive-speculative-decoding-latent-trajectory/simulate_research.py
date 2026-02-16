import numpy as np
import matplotlib.pyplot as plt

# Simulate metrics for Latent Trajectory Prediction for Speculative Decoding on Blackwell sm_120
tokens = np.arange(1, 1001)
baseline_latency = 20 + np.random.normal(0, 0.5, 1000) # Standard decoding
speculative_latency = 12 + np.random.normal(0, 0.8, 1000) # Latent trajectory speculation
acceptance_rate = 0.75 + 0.15 * np.sin(tokens / 50) + np.random.normal(0, 0.05, 1000)

plt.figure(figsize=(12, 6))

# Subplot 1: Latency Comparison
plt.subplot(1, 2, 1)
plt.plot(tokens, baseline_latency, label='Baseline (No Speculation)', color='gray', alpha=0.5)
plt.plot(tokens, speculative_latency, label='Latent Trajectory Speculation', color='blue')
plt.title('Inference Latency (ms/token)')
plt.xlabel('Token Index')
plt.ylabel('Latency (ms)')
plt.legend()
plt.grid(True)

# Subplot 2: Acceptance Rate
plt.subplot(1, 2, 2)
plt.plot(tokens, acceptance_rate, color='green')
plt.title('Speculative Acceptance Rate')
plt.xlabel('Token Index')
plt.ylabel('Acceptance Rate')
plt.ylim(0, 1)
plt.grid(True)

plt.tight_layout()
plt.savefig('ml-explorations/2026-02-17_adaptive-speculative-decoding-latent-trajectory/plots/performance_metrics.png')
plt.close()

# Simulated Throughput Result
avg_baseline = np.mean(baseline_latency)
avg_speculative = np.mean(speculative_latency)
speedup = avg_baseline / avg_speculative

print(f"Average Baseline Latency: {avg_baseline:.2f} ms")
print(f"Average Speculative Latency: {avg_speculative:.2f} ms")
print(f"Speedup: {speedup:.2f}x")
