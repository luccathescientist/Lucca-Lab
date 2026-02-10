import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class CrossModalSteerabilitySim(nn.Module):
    def __init__(self):
        super().__init__()
        # Simulate Qwen2-VL spatial feature map (e.g., 24x24 patches)
        self.visual_features = torch.randn(1, 576, 1024) # [B, N, D]
        # Simulate R1 hidden states (e.g., seq_len 128)
        self.reasoning_states = torch.randn(1, 128, 1024)
        
    def get_steering_weights(self, roi_indices):
        """
        roi_indices: list of indices in the visual feature map to focus on.
        """
        mask = torch.zeros(1, 576, 1)
        mask[0, roi_indices, 0] = 1.0
        # Smoothen mask
        # (In a real scenario, this would be based on spatial proximity)
        return mask

    def steer_attention(self, q, k, v, steering_mask):
        """
        Modified scaled dot-product attention with spatial steering.
        """
        d_k = q.size(-1)
        # Standard attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)
        
        # Inject steering: amplify scores that correlate with the ROI
        # We simulate this by correlating reasoning tokens with visual ROI
        # In reality, this involves cross-modal mapping
        steering_bias = torch.matmul(steering_mask.transpose(-2, -1), k.transpose(-2, -1))
        scores = scores + 0.5 * steering_bias
        
        p_attn = F.softmax(scores, dim=-1)
        return torch.matmul(p_attn, v), p_attn

def run_simulation():
    sim = CrossModalSteerabilitySim()
    
    # Define an ROI (e.g., a "target object" in the center of the visual field)
    roi = list(range(250, 300)) 
    mask = sim.get_steering_weights(roi)
    
    q = torch.randn(1, 128, 1024)
    k = torch.randn(1, 128, 1024)
    v = torch.randn(1, 128, 1024)
    
    # Baseline
    _, baseline_attn = sim.steer_attention(q, k, v, torch.zeros_like(mask))
    # Steered
    _, steered_attn = sim.steer_attention(q, k, v, mask)
    
    # Analysis
    diff = (steered_attn - baseline_attn).detach().numpy()[0]
    
    plt.figure(figsize=(10, 6))
    plt.imshow(diff, aspect='auto', cmap='viridis')
    plt.colorbar(label='Attention Shift')
    plt.title('Cross-Modal Attention Steerability: Steering ROI Impact')
    plt.xlabel('Key Tokens')
    plt.ylabel('Query Tokens')
    plt.savefig('ml-explorations/2026-02-11_cross-modal-attention-steerability/steering_chart.png')
    
    with open('ml-explorations/2026-02-11_cross-modal-attention-steerability/results.txt', 'w') as f:
        f.write(f"Mean attention shift: {np.mean(np.abs(diff))}\n")
        f.write(f"Max attention shift: {np.max(diff)}\n")

if __name__ == "__main__":
    run_simulation()
