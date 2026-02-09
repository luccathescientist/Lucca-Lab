import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np

# Configuration
NUM_EXPERTS = 64
NUM_CLUSTERS = 8
EXPERTS_PER_CLUSTER = 8
D_MODEL = 1024
TOP_K_CLUSTERS = 1
TOP_K_EXPERTS_PER_CLUSTER = 2

class HierarchicalMoERouter(nn.Module):
    def __init__(self, d_model, num_clusters, experts_per_cluster):
        super().__init__()
        self.cluster_router = nn.Linear(d_model, num_clusters)
        self.expert_routers = nn.ModuleList([
            nn.Linear(d_model, experts_per_cluster) for _ in range(num_clusters)
        ])
        
    def forward(self, x):
        # Stage 1: Cluster Routing
        cluster_logits = self.cluster_router(x)
        cluster_probs = torch.softmax(cluster_logits, dim=-1)
        top_cluster_weights, top_cluster_indices = torch.topk(cluster_probs, TOP_K_CLUSTERS, dim=-1)
        
        # Stage 2: Specialist Routing within chosen clusters
        final_expert_indices = []
        final_expert_weights = []
        
        for i in range(TOP_K_CLUSTERS):
            cluster_idx = top_cluster_indices[:, i]
            # In a real implementation, we'd batch this, but for simulation:
            expert_logits = self.expert_routers[cluster_idx[0]](x) # Simplified for timing
            expert_probs = torch.softmax(expert_logits, dim=-1)
            top_expert_weights, top_expert_indices = torch.topk(expert_probs, TOP_K_EXPERTS_PER_CLUSTER, dim=-1)
            
            # Global index = cluster_idx * experts_per_cluster + local_expert_idx
            global_indices = cluster_idx.unsqueeze(-1) * experts_per_cluster + top_expert_indices
            
            final_expert_indices.append(global_indices)
            final_expert_weights.append(top_expert_weights * top_cluster_weights[:, i:i+1])
            
        return torch.cat(final_expert_indices, dim=-1), torch.cat(final_weights, dim=-1)

def benchmark_routing():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Benchmarking on {device}...")
    
    d_model = D_MODEL
    batch_size = 1024
    seq_len = 128
    
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    
    # Standard Flat Router (1-stage)
    flat_router = nn.Linear(d_model, NUM_EXPERTS).to(device)
    
    # Hierarchical Router
    h_router = HierarchicalMoERouter(d_model, NUM_CLUSTERS, EXPERTS_PER_CLUSTER).to(device)
    
    # Warmup
    for _ in range(10):
        _ = flat_router(x)
        _ = h_router.cluster_router(x)
        
    # Benchmark Flat
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        logits = flat_router(x)
        _, _ = torch.topk(logits, TOP_K_CLUSTERS * TOP_K_EXPERTS_PER_CLUSTER, dim=-1)
    torch.cuda.synchronize()
    flat_time = (time.time() - start) / 100
    
    # Benchmark Hierarchical (Total logic)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        # Stage 1
        c_logits = h_router.cluster_router(x)
        _, c_indices = torch.topk(c_logits, TOP_K_CLUSTERS, dim=-1)
        # Stage 2 (Simulated gated activation)
        for i in range(TOP_K_CLUSTERS):
            _ = h_router.expert_routers[0](x) # Benchmarking specific router overhead
    torch.cuda.synchronize()
    h_time = (time.time() - start) / 100
    
    print(f"Flat Routing Time: {flat_time*1000:.4f} ms")
    print(f"Hierarchical Routing Time: {h_time*1000:.4f} ms")
    
    # Visualization
    labels = ['Flat (64 Experts)', 'Hierarchical (8x8 Clusters)']
    times = [flat_time * 1000, h_time * 1000]
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, times, color=['#00cfcf', '#ff00ff'])
    plt.ylabel('Latency (ms)')
    plt.title('MoE Routing Latency: Flat vs Hierarchical (Blackwell Simulation)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('routing_benchmark.png')
    
if __name__ == "__main__":
    benchmark_routing()
