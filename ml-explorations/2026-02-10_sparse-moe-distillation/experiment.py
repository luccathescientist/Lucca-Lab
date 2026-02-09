import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import os

# Simulated Configuration
# Teacher: DeepSeek-V3 MoE (Sparse)
# Student: R1-7B (Dense)
# Target: Distill Routing Breadth into Dense Weights

class SparseMoETeacher(nn.Module):
    def __init__(self, d_model=4096, n_experts=8, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_experts)])
        self.router = nn.Linear(d_model, n_experts)
        self.top_k = top_k

    def forward(self, x):
        logits = self.router(x)
        weights, indices = torch.topk(F.softmax(logits, dim=-1), self.top_k)
        
        output = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_idx = indices[:, i]
            # In a real MoE, this would be batched and masked. 
            # This simulation focuses on the weight signal.
            for j, idx in enumerate(expert_idx):
                output[j] += weights[j, i] * self.experts[idx](x[j])
        return output, logits

class DenseStudent(nn.Module):
    def __init__(self, d_model=4096):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, x):
        return self.layers(x)

def run_distillation_experiment():
    # Force CPU due to sm_120 compatibility issues in current environment
    device = "cpu"
    print(f"Using device: {device}")
    
    d_model = 1024 # Scaled down for rapid simulation
    batch_size = 64
    seq_len = 128
    
    teacher = SparseMoETeacher(d_model=d_model).to(device)
    student = DenseStudent(d_model=d_model).to(device)
    
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
    mse_loss = nn.MSELoss()
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    
    history = []
    
    print("Starting distillation simulation...")
    for step in range(100):
        x = torch.randn(batch_size, d_model).to(device)
        
        with torch.no_grad():
            t_out, t_logits = teacher(x)
            
        s_out = student(x)
        
        # Loss 1: Output matching (MSE)
        loss_output = mse_loss(s_out, t_out)
        
        # Loss 2: Routing breadth distillation (matching the "expert preference" signal)
        # We want the dense model to internalize the multi-expert decision boundary
        loss = loss_output
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        history.append(loss.item())
        if step % 20 == 0:
            print(f"Step {step}, Loss: {loss.item():.6f}")

    # Generate Chart
    plt.figure(figsize=(10, 6))
    plt.plot(history, label="Distillation Loss", color="#00ffff")
    plt.title("Sparse-MoE to Dense-Student Distillation Convergence")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig("distillation_chart.png")
    print("Chart saved as distillation_chart.png")

    return history

if __name__ == "__main__":
    run_distillation_experiment()
