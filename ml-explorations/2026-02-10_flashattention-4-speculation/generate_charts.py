import matplotlib.pyplot as plt

# Comparison: FlashAttention-2 vs FlashAttention-3 vs FA4 (Speculative)
versions = ['FA2', 'FA3', 'FA4 (Spec)']
latency = [0.15, 0.11, 0.082]  # ms

plt.figure(figsize=(10, 6))
plt.bar(versions, latency, color=['gray', 'blue', 'purple'])
plt.title('FlashAttention Latency Projection on Blackwell (sm_120)')
plt.ylabel('Latency (ms)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('ml-explorations/2026-02-10_flashattention-4-speculation/latency_projection.png')
