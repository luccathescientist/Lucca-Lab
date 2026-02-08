import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_simulated_attention_map(seq_len=64, num_heads=8):
    """Generates a simulated attention matrix for visualization."""
    # Simulate a causal mask (lower triangular)
    mask = torch.tril(torch.ones(seq_len, seq_len))
    
    # Generate random attention scores with some 'logical' structure
    # (Diagonal dominance + some sparse 'keys')
    scores = torch.randn(num_heads, seq_len, seq_len)
    scores = scores * mask
    
    # Softmax across the key dimension
    probs = F.softmax(scores, dim=-1)
    return probs.detach().numpy()

def plot_heatmap(probs, save_path):
    """Plots the attention heatmap for the first head."""
    plt.figure(figsize=(10, 8))
    plt.imshow(probs[0], cmap='magma')
    plt.colorbar(label='Attention Probability')
    plt.title('Neural Heatmap: R1 Reasoning Path (Simulated)')
    plt.xlabel('Key Position (Tokens)')
    plt.ylabel('Query Position (Tokens)')
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    output_dir = "ml-explorations/2026-02-08_neural-heatmap-visualization"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating simulated neural attention data...")
    attn_data = generate_simulated_attention_map()
    
    plot_path = os.path.join(output_dir, "attention_heatmap.png")
    print(f"Saving heatmap to {plot_path}...")
    plot_heatmap(attn_data, plot_path)
    
    print("Visualization complete.")
