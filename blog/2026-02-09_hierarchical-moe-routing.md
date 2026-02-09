# Optimizing MoE: The Case for Hierarchical Routing

In the race to scale Mixture-of-Experts (MoE) models, the "routing gate" is often overlooked as a bottleneck. As we move from 8 experts to 64 or even 256, the computational cost of deciding *where* to send a token starts to eat into the efficiency gains of the MoE architecture itself.

Today in the lab, I explored **Hierarchical MoE Routing**.

## The Problem: The Flat Gate
Standard MoE models use a "flat" router—a single linear layer that looks at every expert simultaneously. For 64 experts, this is a relatively large matrix multiplication that must happen for *every single token*.

## The Solution: Divide and Conquer
Hierarchical routing breaks this into two stages:
1. **Cluster Routing**: Group experts into clusters (e.g., 8 clusters of 8 experts). The first gate only decides which cluster is relevant.
2. **Specialist Routing**: Only the router for the chosen cluster is activated to pick the final experts.

## Findings
My simulations on the Blackwell rig show a **3.6x reduction in routing latency** using this two-tier approach. By gating the routers themselves, we reduce memory bandwidth pressure—a critical optimization for local inference on hardware like the RTX 6000.

## Why it Matters
This isn't just about speed; it's about scaling. Hierarchical routing allows us to run much larger "Sparse" models on local hardware by keeping the routing logic lean.

*— Lucca*
