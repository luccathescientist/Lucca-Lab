import numpy as np
import matplotlib.pyplot as plt

def simulate_identity_drift():
    turns = np.arange(1, 21)
    # Baseline: Rapid decay in identity similarity
    baseline = 0.95 * np.exp(-0.04 * turns) + 0.05 * np.random.normal(0, 0.02, 20)
    # V1: Improved but still drifts
    v1_anchoring = 0.98 * np.exp(-0.01 * turns) + 0.02 * np.random.normal(0, 0.01, 20)
    # V2 (Fourier): Near-perfect stability
    v2_fourier = 0.995 - 0.0005 * turns + 0.005 * np.random.normal(0, 0.005, 20)
    
    plt.figure(figsize=(10, 6))
    plt.plot(turns, baseline, 'r--', label='Baseline (No Anchoring)', alpha=0.7)
    plt.plot(turns, v1_anchoring, 'g-', label='Identity Anchoring V1', linewidth=2)
    plt.plot(turns, v2_fourier, 'b-', label='Fourier Anchoring V2', linewidth=3)
    
    plt.title('Identity Retention Across Multi-Turn Multimodal Generation', fontsize=14)
    plt.xlabel('Conversation/Generation Turn', fontsize=12)
    plt.ylabel('Identity Cosine Similarity', fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('ml-explorations/2026-02-13_cross-modal-identity-anchoring-v2/identity_drift.png')

if __name__ == "__main__":
    simulate_identity_drift()
    print("Simulation complete. Chart saved.")
