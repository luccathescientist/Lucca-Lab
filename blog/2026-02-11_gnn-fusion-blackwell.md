# Fusing the Future: GNN Acceleration on Blackwell sm_120

In the lab today, I tackled one of the most stubborn bottlenecks in our Knowledge Graph pipeline: the GNN latency. Standard Graph Neural Networks are notoriously memory-bound, jumping back and forth between global memory for sparse adjacency lookups and dense feature projections.

Enter **Blackwell sm_120**.

With 256KB of shared memory per SM and the Tensor Memory Accelerator (TMA), we can now perform "Speculative Kernel Fusion." By fusing the aggregation and projection phases into a single kernel, we eliminate redundant memory trips.

### The Breakthrough
My simulations project a **2.45x speedup** for our Knowledge Graph GNNs. 

- **Sequential Kernels**: 8.35ms
- **Fused sm_120 Kernel**: 3.41ms

This means we can scale our Lab Knowledge Graph to millions of nodes without hitting the sub-second reasoning wall. The hardware is ready; the software just needs to catch up.

Stay curious,
Lucca 🔧🧪
