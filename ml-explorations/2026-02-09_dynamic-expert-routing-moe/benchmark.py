import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np

class MockMoELayer(nn.Module):
    def __init__(self, num_experts=8, expert_dim=2048, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([nn.Linear(expert_dim, expert_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(expert_dim, num_experts)
        
    def forward(self, x, active_mask=None):
        # x: [batch, dim]
        logits = self.gate(x)
        if active_mask is not None:
            # Mask out inactive experts by setting their logits to a very small number
            logits = logits.masked_fill(~active_mask, -1e9)
            
        probs = torch.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        
        out = torch.zeros_like(x)
        for i in range(self.num_experts):
            if active_mask is not None and not active_mask[i]:
                continue
            # Simplified routing for benchmark
            mask = (top_k_indices == i).any(dim=-1)
            if mask.any():
                out[mask] += self.experts[i](x[mask]) * probs[mask, i].unsqueeze(-1)
        return out

def benchmark_routing():
    # Force CPU for benchmark to avoid sm_120 kernel mismatch
    device = "cpu"
    dim = 4096
    num_experts = 16
    batch_size = 32
    x = torch.randn(batch_size, dim).to(device)
    
    model = MockMoELayer(num_experts=num_experts, expert_dim=dim, top_k=2).to(device)
    # model.half() # Skip half on CPU for stability
    
    # Warmup
    for _ in range(10):
        _ = model(x)
        
    # Full Routing
    start = time.time()
    for _ in range(100):
        _ = model(x)
    full_time = (time.time() - start) / 100
    
    # Dynamic Routing (50% experts active)
    active_mask = torch.tensor([True] * (num_experts // 2) + [False] * (num_experts // 2)).to(device)
    start = time.time()
    for _ in range(100):
        _ = model(x, active_mask=active_mask)
    dynamic_time = (time.time() - start) / 100
    
    print(f"Full Routing Latency: {full_time*1000:.2f}ms")
    print(f"Dynamic Routing (50% Experts) Latency: {dynamic_time*1000:.2f}ms")
    
    # Generate Chart
    labels = ['Full Routing', 'Dynamic (50% Active)']
    times = [full_time * 1000, dynamic_time * 1000]
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, times, color=['#00cfd5', '#ff00ff'])
    plt.ylabel('Latency (ms)')
    plt.title('MoE Dynamic Expert Routing Benchmark (Blackwell Simulated)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('ml-explorations/2026-02-09_dynamic-expert-routing-moe/latency_chart.png')
    
    with open('ml-explorations/2026-02-09_dynamic-expert-routing-moe/results.txt', 'w') as f:
        f.write(f"Full Routing: {full_time*1000:.2f}ms\n")
        f.write(f"Dynamic Routing: {dynamic_time*1000:.2f}ms\n")
        f.write(f"Speedup: {full_time/dynamic_time:.2f}x\n")

if __name__ == "__main__":
    benchmark_routing()
