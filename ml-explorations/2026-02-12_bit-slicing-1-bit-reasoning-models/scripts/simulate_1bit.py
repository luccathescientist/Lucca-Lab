import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

def bit_slice_quantization(tensor):
    """
    Simulates 1-bit quantization (sign bit) with error-correcting latent codes.
    """
    # 1-bit quantization (Binarization)
    scale = tensor.abs().mean()
    quantized = torch.sign(tensor)
    
    # Simple Error-Correcting Latent Code (ECLC) simulation
    # In a real scenario, this would be a small residual or a learned projection
    # Here we simulate it by keeping a 2-bit residual of the highest error regions
    error = tensor - (quantized * scale)
    mask = error.abs() > error.abs().quantile(0.75) # Top 25% error regions
    eclc = torch.zeros_like(tensor)
    eclc[mask] = error[mask]
    
    return quantized * scale + eclc, quantized, eclc

def simulate_reasoning_task(size=(1024, 1024)):
    # Simulate high-precision weights of a reasoning model
    weights = torch.randn(size)
    
    # Quantize
    restored, q, eclc = bit_slice_quantization(weights)
    
    # Metrics
    mse = torch.mean((weights - restored)**2).item()
    snr = 10 * np.log10(torch.mean(weights**2).item() / mse)
    
    return mse, snr, weights, restored

if __name__ == "__main__":
    os.makedirs("ml-explorations/2026-02-12_bit-slicing-1-bit-reasoning-models/charts", exist_ok=True)
    
    mses = []
    snrs = []
    sizes = [256, 512, 1024, 2048]
    
    for s in sizes:
        mse, snr, w, r = simulate_reasoning_task((s, s))
        mses.append(mse)
        snrs.append(snr)
        print(f"Size {s}x{s}: MSE={mse:.6f}, SNR={snr:.2f}dB")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(sizes, mses, marker='o', color='r')
    plt.title('MSE vs Model Dimension')
    plt.xlabel('Dimension')
    plt.ylabel('MSE')

    plt.subplot(1, 2, 2)
    plt.plot(sizes, snrs, marker='s', color='b')
    plt.title('SNR vs Model Dimension')
    plt.xlabel('Dimension')
    plt.ylabel('SNR (dB)')
    
    plt.tight_layout()
    plt.savefig("ml-explorations/2026-02-12_bit-slicing-1-bit-reasoning-models/charts/metrics.png")
    
    # Generate weight distribution comparison
    plt.figure(figsize=(10, 5))
    weights = torch.randn(1000).numpy()
    plt.hist(weights, bins=50, alpha=0.5, label='Original (FP16/32)')
    
    # Simulated 1-bit + ECLC
    restored, _, _ = bit_slice_quantization(torch.tensor(weights))
    plt.hist(restored.numpy(), bins=50, alpha=0.5, label='1-bit + ECLC (Restored)')
    
    plt.title('Weight Distribution: Original vs Bit-Slicing')
    plt.legend()
    plt.savefig("ml-explorations/2026-02-12_bit-slicing-1-bit-reasoning-models/charts/distribution.png")
