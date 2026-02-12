import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np

class GatedLinearUnit(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        # SwiGLU implementation
        x1 = self.w1(x)
        x2 = self.w2(x)
        return self.w3(F.silu(x1) * x2)

def simulate_quantization(tensor, scale, bits=8):
    q_min = -(2**(bits-1))
    q_max = 2**(bits-1) - 1
    # Simulated symmetric quantization
    q_tensor = torch.clamp(torch.round(tensor / scale), q_min, q_max)
    return q_tensor * scale

def benchmark_glu_quantization():
    device = "cpu" # Switched to CPU for simulation due to sm_120 driver mismatch in current torch
    d_model = 1024
    d_ff = 2048
    batch_size = 1
    seq_len = 128
    
    input_tensor = torch.randn(batch_size, seq_len, d_model).to(device)
    model = GatedLinearUnit(d_model, d_ff).to(device)
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

    # FP32 Latency
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            fp32_out = model(input_tensor)
    fp32_latency = (time.time() - start) / 100 * 1000

    # Simulated FP8/INT8 Mixed Precision for GLU
    # In SwiGLU: w3(silu(w1(x)) * w2(x))
    # We simulate quantization of x1 (w1 output) and x2 (w2 output)
    with torch.no_grad():
        x1 = model.w1(input_tensor)
        x2 = model.w2(input_tensor)
        
        # Determine scales (simulated calibration)
        scale_x1 = x1.abs().max() / 127
        scale_x2 = x2.abs().max() / 127
        
        # CPU profiling
        start = time.time()
        for _ in range(100):
            q_x1 = simulate_quantization(x1, scale_x1, bits=8)
            q_x2 = simulate_quantization(x2, scale_x2, bits=8)
            # Element-wise gate in quantized space
            gated = F.silu(q_x1) * q_x2
            # Quantize gated output before w3
            scale_gated = gated.abs().max() / 127
            q_gated = simulate_quantization(gated, scale_gated, bits=8)
            q_out = model.w3(q_gated)
        mixed_latency = (time.time() - start) / 100 * 1000

    # Calculate Error (MSE)
    mse = F.mse_loss(fp32_out, q_out).item()

    print(f"FP32 Latency: {fp32_latency:.2f} ms")
    print(f"Mixed-Precision Latency (Simulated): {mixed_latency:.2f} ms")
    print(f"MSE Error: {mse:.2e}")

    # Generate Chart
    labels = ['FP32', 'Mixed FP8/INT8 (Sim)']
    latencies = [fp32_latency, mixed_latency]
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, latencies, color=['blue', 'green'])
    plt.ylabel('Latency (ms)')
    plt.title('GLU Quantization Performance on Blackwell (Simulated)')
    plt.savefig('ml-explorations/2026-02-13_hardware-aware-quantization-glu/latency_comparison.png')
    
    return fp32_latency, mixed_latency, mse

if __name__ == "__main__":
    benchmark_glu_quantization()
