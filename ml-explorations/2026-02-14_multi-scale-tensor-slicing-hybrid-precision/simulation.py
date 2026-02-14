import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

class MultiScaleTensorSlicer:
    """
    Simulates Multi-Scale Tensor Slicing for Hybrid Precision on Blackwell sm_120.
    Slices weights into a 'Base' (low precision, high magnitude) and 'Residual' (higher precision, low magnitude) component.
    """
    def __init__(self, weight, base_bits=4, residual_bits=8):
        self.weight = weight
        self.base_bits = base_bits
        self.residual_bits = residual_bits
        
    def slice_tensors(self):
        # Simulate slicing logic
        max_val = self.weight.abs().max()
        scale_base = max_val / (2**(self.base_bits-1) - 1)
        
        # Base component (INT4 simulated)
        base_quant = torch.round(self.weight / scale_base).clamp(-(2**(self.base_bits-1)), 2**(self.base_bits-1) - 1)
        base_dequant = base_quant * scale_base
        
        # Residual component (difference at FP8/INT8 precision)
        residual = self.weight - base_dequant
        scale_res = residual.abs().max() / (2**(self.residual_bits-1) - 1) if residual.abs().max() > 0 else 1.0
        res_quant = torch.round(residual / scale_res).clamp(-(2**(self.residual_bits-1)), 2**(self.residual_bits-1) - 1)
        res_dequant = res_quant * scale_res
        
        return base_dequant, res_dequant

def run_simulation():
    # Large weight matrix simulation (e.g., 4096 x 4096)
    size = 4096
    weight = torch.randn(size, size)
    
    slicer = MultiScaleTensorSlicer(weight)
    
    start_time = time.time()
    base, residual = slicer.slice_tensors()
    end_time = time.time()
    
    reconstructed = base + residual
    mse = torch.mean((weight - reconstructed)**2).item()
    
    print(f"Simulation Complete.")
    print(f"MSE: {mse:.8e}")
    print(f"Processing Time: {(end_time - start_time)*1000:.2f} ms")
    
    # Generate charts
    plt.figure(figsize=(10, 6))
    plt.hist(weight.flatten().numpy(), bins=100, alpha=0.5, label='Original')
    plt.hist(base.flatten().numpy(), bins=100, alpha=0.5, label='Base (Slicing)')
    plt.hist(residual.flatten().numpy(), bins=100, alpha=0.5, label='Residual')
    plt.title('Weight Distribution: Multi-Scale Tensor Slicing')
    plt.legend()
    plt.savefig('ml-explorations/2026-02-14_multi-scale-tensor-slicing-hybrid-precision/weight_distribution.png')
    
    # Precision Improvement chart
    precisions = [4, 6, 8]
    mses = []
    for p in precisions:
        s = MultiScaleTensorSlicer(weight, base_bits=p)
        b, r = s.slice_tensors()
        mses.append(torch.mean((weight - (b + r))**2).item())
        
    plt.figure(figsize=(10, 6))
    plt.plot(precisions, mses, marker='o')
    plt.yscale('log')
    plt.xlabel('Base Bits')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Reconstruction Error vs. Slicing Precision')
    plt.grid(True)
    plt.savefig('ml-explorations/2026-02-14_multi-scale-tensor-slicing-hybrid-precision/precision_curve.png')

if __name__ == "__main__":
    run_simulation()
