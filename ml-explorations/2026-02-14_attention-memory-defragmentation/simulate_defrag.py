import numpy as np
import matplotlib.pyplot as plt
import os

# Simulated Data for Attention-Based Memory Defragmentation
# Comparing Standard PagedAttention vs. Attention-Decay Aware Defragmentation

def simulate_vram_fragmentation():
    time_steps = np.arange(0, 100, 1)
    
    # Standard PagedAttention: Linear-ish growth in fragmentation until OOM/Flush
    standard_fragmentation = 0.5 * time_steps + np.random.normal(0, 2, len(time_steps))
    standard_fragmentation = np.clip(standard_fragmentation, 0, 95)
    
    # Attention-Decay Aware: Proactive defrag keeps it stable
    # Logic: As attention weights decay, non-critical tokens are evicted or compressed
    decay_aware_fragmentation = 15 + 5 * np.sin(time_steps / 5) + np.random.normal(0, 1, len(time_steps))
    decay_aware_fragmentation = np.clip(decay_aware_fragmentation, 5, 30)

    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, standard_fragmentation, label='Standard PagedAttention', color='red', linestyle='--')
    plt.plot(time_steps, decay_aware_fragmentation, label='Attention-Decay Aware Defragmentation', color='green', linewidth=2)
    
    plt.title('VRAM Fragmentation Over Time (Simulated)')
    plt.xlabel('Inference Steps (Multi-Turn Context)')
    plt.ylabel('VRAM Fragmentation (%)')
    plt.axhline(y=85, color='orange', linestyle=':', label='OOM Danger Zone')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = 'ml-explorations/2026-02-14_attention-memory-defragmentation/fragmentation_chart.png'
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")

if __name__ == "__main__":
    simulate_vram_fragmentation()
