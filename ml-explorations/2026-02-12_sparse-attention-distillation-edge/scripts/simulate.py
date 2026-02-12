import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import matplotlib.pyplot as plt
import numpy as np

class SparseAttention(nn.Module):
    def __init__(self, d_model, n_heads, sparsity=0.9):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.sparsity = sparsity
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Simulate Sparse Attention (Top-K)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (D // self.n_heads)**0.5
        
        # Create mask for sparsity
        k_val = int(L * (1 - self.sparsity))
        if k_val < 1: k_val = 1
        
        topk_vals, _ = torch.topk(attn_scores, k_val, dim=-1)
        min_val = topk_vals[..., -1:]
        mask = (attn_scores >= min_val).float()
        
        # Apply mask and softmax
        attn_probs = F.softmax(attn_scores.masked_fill(mask == 0, float('-inf')), dim=-1)
        out = torch.matmul(attn_probs, v)
        out = out.permute(0, 2, 1, 3).reshape(B, L, D)
        return self.out(out), attn_probs

class DenseAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (D // self.n_heads)**0.5
        attn_probs = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_probs, v)
        out = out.permute(0, 2, 1, 3).reshape(B, L, D)
        return self.out(out), attn_probs

def run_simulation():
    device = "cpu"
    d_model = 256
    n_heads = 4
    seq_len = 512
    sparsity = 0.95
    
    teacher = SparseAttention(d_model, n_heads, sparsity).to(device).eval()
    student = DenseAttention(d_model, n_heads).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
    
    x = torch.randn(8, seq_len, d_model).to(device)
    
    # Benchmarking
    # torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(10): teacher(x)
    # torch.cuda.synchronize()
    teacher_time = (time.time() - start) / 10
    
    # Distillation Loop (Simplified)
    losses = []
    print("Starting distillation...")
    for i in range(100):
        optimizer.zero_grad()
        with torch.no_grad():
            t_out, t_attn = teacher(x)
        s_out, s_attn = student(x)
        
        # Loss: KL Divergence on Attention + MSE on Output
        loss_attn = F.mse_loss(s_attn, t_attn) 
        loss_out = F.mse_loss(s_out, t_out)
        loss = loss_out + 10 * loss_attn
        
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if i % 10 == 0:
            print(f"Step {i}, Loss: {loss.item():.6f}")

    # Results
    # torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(10): student(x)
    # torch.cuda.synchronize()
    student_time = (time.time() - start) / 10

    results = {
        "teacher_latency_ms": teacher_time * 1000,
        "student_latency_ms": student_time * 1000,
        "final_loss": losses[-1],
        "sparsity_level": sparsity,
        "throughput_gain": teacher_time / student_time if student_time > 0 else 0
    }
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Distillation Loss (Sparse -> Dense)")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig("loss_curve.png")
    
    print(f"Simulation complete. Student Latency: {results['student_latency_ms']:.2f}ms")

if __name__ == "__main__":
    run_simulation()
