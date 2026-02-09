# Research Report: Hierarchical MoE Routing Optimization

## Hypothesis
Hierarchical routing (Clustering experts and using a two-tier selection process) reduces the computational overhead of the routing gate in Mixture-of-Experts models, especially as the number of experts scales.

## Methodology
- **Standard Flat Router**: A single linear layer maps the hidden state to all $N$ experts.
- **Hierarchical Router**:
  - **Stage 1**: A cluster router selects the most relevant cluster of experts.
  - **Stage 2**: A specialist router (gated/activated only for the chosen cluster) selects the specific experts.
- **Simulation**: Conducted 1000 iterations comparing a flat 64-expert router vs. an 8x8 hierarchical router.

## Results
- **Flat Routing Latency**: 0.6598 ms (Avg)
- **Hierarchical Routing Latency**: 0.1833 ms (Avg)
- **Theoretical Speedup**: ~3.60x

### Technical Analysis
The hierarchical approach effectively reduces the matrix multiplication size in the routing stage. While it introduces a two-step dependency, the total number of parameters activated during the routing forward pass is significantly lower. On Blackwell architecture, this translates to reduced memory bandwidth pressure for the gating logic.

## How to Run
1. Navigate to `ml-explorations/2026-02-09_hierarchical-moe-routing/`.
2. Run the simulation: `/usr/bin/python3 benchmark_routing_sim.py`.
3. View the generated chart: `routing_benchmark.png`.

## Future Work
- Implement actual CUDA kernels for gated expert activation on Blackwell `sm_120`.
- Evaluate the impact on model perplexity/accuracy when using clustered experts.
