import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Simulate Blackwell FP8 Noise and Hardware-Aware DPO
class BlackwellFP8Simulator:
    def __init__(self, noise_scale=0.05):
        self.noise_scale = noise_scale

    def apply_quantization_noise(self, tensor):
        # Simulate FP8 quantization error as additive Gaussian noise
        noise = torch.randn_like(tensor) * self.noise_scale
        return tensor + noise

def dpo_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta=0.1):
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    logits = pi_logratios - ref_logratios
    return -F.logsigmoid(beta * logits).mean()

def run_experiment():
    print("Initializing Hardware-Aware DPO Simulation...")
    
    # Mock data: log probabilities for chosen and rejected responses
    batch_size = 64
    beta = 0.1
    
    # Reference model logps (fixed)
    ref_chosen = torch.randn(batch_size)
    ref_rejected = torch.randn(batch_size)
    
    # Policy model logps (initial)
    policy_chosen = ref_chosen.clone() + torch.randn(batch_size) * 0.1
    policy_rejected = ref_rejected.clone() + torch.randn(batch_size) * 0.1
    
    noise_levels = np.linspace(0, 0.2, 10)
    standard_losses = []
    hardware_aware_losses = []
    
    simulator = BlackwellFP8Simulator()
    
    for noise in noise_levels:
        simulator.noise_scale = noise
        
        # Standard DPO (unaware of noise)
        noisy_chosen = simulator.apply_quantization_noise(policy_chosen)
        noisy_rejected = simulator.apply_quantization_noise(policy_rejected)
        loss_std = dpo_loss(noisy_chosen, noisy_rejected, ref_chosen, ref_rejected, beta)
        standard_losses.append(loss_std.item())
        
        # Hardware-Aware DPO (Adaptive Beta / Noise Penalty)
        # Hypothesis: Increasing Beta or adding a regularization term for high-variance (noisy) updates improves stability.
        adaptive_beta = beta * (1 + noise * 5) # Heuristic for mitigation
        loss_aware = dpo_loss(noisy_chosen, noisy_rejected, ref_chosen, ref_rejected, adaptive_beta)
        hardware_aware_losses.append(loss_aware.item())

    # Generate Chart
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, standard_losses, label='Standard DPO (FP8 Unaware)', marker='o')
    plt.plot(noise_levels, hardware_aware_losses, label='Hardware-Aware DPO (Adaptive Beta)', marker='s')
    plt.title('DPO Stability under Blackwell FP8 Quantization Noise')
    plt.xlabel('Quantization Noise Scale (Simulated FP8)')
    plt.ylabel('Negative Log-Likelihood Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('dpo_fp8_stability.png')
    print("Experiment complete. Chart saved to dpo_fp8_stability.png")

if __name__ == "__main__":
    run_experiment()
