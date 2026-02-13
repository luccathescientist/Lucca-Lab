import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

def fourier_encode(x, num_frequencies=16):
    """Encodes a latent vector into Fourier space."""
    frequencies = 2.0 ** torch.arange(num_frequencies, device=x.device) * np.pi
    encoded = []
    for freq in frequencies:
        encoded.append(torch.sin(x * freq))
        encoded.append(torch.cos(x * freq))
    return torch.cat(encoded, dim=-1)

class IdentityAnchorV2(nn.Module):
    def __init__(self, latent_dim=1024, fourier_dim=16):
        super().__init__()
        self.anchor = nn.Parameter(torch.randn(1, latent_dim))
        # fourier_encode returns latent_dim * fourier_dim * 2
        self.projection = nn.Linear(latent_dim * fourier_dim * 2, latent_dim)
        
    def forward(self, x, alpha=0.1):
        # x is the input latent trajectory
        encoded_anchor = fourier_encode(self.anchor)
        steer_signal = self.projection(encoded_anchor)
        # Residual steering
        return x + alpha * (steer_signal - x)

def simulate_drift(steps=50, latent_dim=1024, noise_level=0.05):
    """Simulates character drift over video frames or turns."""
    trajectory = [torch.randn(1, latent_dim)]
    for _ in range(steps - 1):
        noise = torch.randn(1, latent_dim) * noise_level
        trajectory.append(trajectory[-1] + noise)
    return torch.stack(trajectory)

def run_experiment():
    device = "cpu"
    latent_dim = 1024
    steps = 100
    
    # Baseline: Drift without anchoring
    drifted_trajectory = simulate_drift(steps=steps, latent_dim=latent_dim).to(device)
    
    # V2: Anchoring with Fourier steering
    anchor_v2 = IdentityAnchorV2(latent_dim=latent_dim).to(device)
    
    start_time = time.time()
    anchored_trajectory = []
    current_latent = drifted_trajectory[0]
    for i in range(steps):
        current_latent = anchor_v2(current_latent, alpha=0.15)
        # Re-inject some drift noise
        current_latent = current_latent + torch.randn(1, latent_dim).to(device) * 0.02
        anchored_trajectory.append(current_latent)
    end_time = time.time()
    
    anchored_trajectory = torch.stack(anchored_trajectory)
    
    # Calculate Cosine Similarity to the original identity
    orig = drifted_trajectory[0]
    
    sim_baseline = torch.nn.functional.cosine_similarity(drifted_trajectory.squeeze(), orig, dim=-1).cpu().numpy()
    sim_anchored = torch.nn.functional.cosine_similarity(anchored_trajectory.squeeze(), orig, dim=-1).detach().cpu().numpy()
    
    # Latency check
    latency = (end_time - start_time) / steps * 1000 # ms
    print(f"Average steering latency: {latency:.4f} ms")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(sim_baseline, label='Baseline (Drift)', color='red', linestyle='--')
    plt.plot(sim_anchored, label='V2 Anchoring (Fourier)', color='cyan')
    plt.title('Character Identity Persistence: Baseline vs Fourier Anchoring V2')
    plt.xlabel('Frame / Turn')
    plt.ylabel('Cosine Similarity to Identity Anchor')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('ml-explorations/2026-02-14_cross-modal-identity-anchoring-v2/results.png')
    
    # Save statistics
    with open('ml-explorations/2026-02-14_cross-modal-identity-anchoring-v2/stats.txt', 'w') as f:
        f.write(f"Latency: {latency:.4f} ms\n")
        f.write(f"Final Similarity (Baseline): {sim_baseline[-1]:.4f}\n")
        f.write(f"Final Similarity (Anchored): {sim_anchored[-1]:.4f}\n")

if __name__ == "__main__":
    run_experiment()
