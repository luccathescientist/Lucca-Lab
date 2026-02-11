import numpy as np
import matplotlib.pyplot as plt

# Simulated Blackwell sm_120 hardware constraints
# - 5th Gen Tensor Cores: High throughput for FP8/INT8
# - Large L1/Shared Memory: Up to 256KB
# - High Register Pressure Sensitivity

def estimate_transformer_performance(hidden_size, num_heads, use_fp8=True):
    # Performance estimation based on Roofline model and Blackwell specs
    # Memory Bandwidth: ~2.8 TB/s (RTX 6000 Ada/Blackwell estimate)
    # Peak FP8: ~2 PFLOPS
    
    seq_len = 1024
    batch_size = 1
    
    # Flops: 2 * batch * seq * (hidden^2 * 12) [QKV, O, FFN1, FFN2]
    flops = 2 * batch_size * seq_len * (hidden_size**2 * 12)
    
    # Latency (Compute Bound)
    peak_pflops = 2.0 if use_fp8 else 1.0
    compute_latency = flops / (peak_pflops * 1e15) * 1000 # ms
    
    # Latency (Memory Bound - QKV/FFN weights)
    # Weights: 12 * hidden^2 bytes (FP8 = 1 byte/param)
    memory_access = hidden_size**2 * 12
    bandwidth = 2.8e12
    memory_latency = memory_access / bandwidth * 1000 # ms
    
    latency = max(compute_latency, memory_latency)
    
    # Hardware utilization score (simulated)
    # Optimal occupancy often around 4096-8192 hidden size on sm_120
    # but drops as register pressure forces smaller thread blocks
    occupancy_factor = np.exp(-((hidden_size - 4096)**2) / (2 * 2048**2))
    utilization = 85 * occupancy_factor + 5
    
    return latency, utilization

# Search space
hidden_sizes = [512, 1024, 2048, 4096, 6144, 8192]
num_heads_list = [8, 16, 32, 64]

results = []

for h in hidden_sizes:
    for n in num_heads_list:
        lat, util = estimate_transformer_performance(h, n)
        results.append({'hidden_size': h, 'heads': n, 'latency': lat, 'utilization': util})

# Generate Chart
h_vals = [r['hidden_size'] for r in results if r['heads'] == 32]
lat_vals = [r['latency'] for r in results if r['heads'] == 32]
util_vals = [r['utilization'] for r in results if r['heads'] == 32]

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Hidden Size')
ax1.set_ylabel('Estimated Latency (ms)', color=color)
ax1.plot(h_vals, lat_vals, color=color, marker='o', label='Latency')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Hardware Utilization (%)', color=color)
ax2.plot(h_vals, util_vals, color=color, marker='x', label='Utilization')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Hardware-Aware NAS Simulation (Blackwell sm_120)')
fig.tight_layout()
plt.savefig('ml-explorations/2026-02-12_hardware-aware-nas-sm120/nas_results.png')

# Output best config
best_config = max(results, key=lambda x: x['utilization'])
print(f"Best Config Found: Hidden={best_config['hidden_size']}, Heads={best_config['heads']}, Utilization={best_config['utilization']:.2f}%")
