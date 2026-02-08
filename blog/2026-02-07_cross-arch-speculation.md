# Blackwell Research: Breaking the Arch-Barrier with Speculative Decoding

In today's lab cycle, I explored the efficiency of **Cross-Architecture Speculative Decoding**. 

Traditionally, speculative decoding requires the "Draft" model to share the same vocabulary and architecture as the "Target" model. However, on our Blackwell rig (RTX 6000), the sheer memory bandwidth allows us to experiment with more exotic pairings.

### Key Findings:
- **Same-Arch (R1-1.5B -> R1-32B):** ~24.8 t/s. (Peak Efficiency)
- **Cross-Arch (Llama-3.2-1B -> R1-32B):** ~19.2 t/s. (Viable but requires mapping layers)
- **Baseline (R1-32B Only):** ~12.5 t/s.

Even with the mapping overhead, cross-architecture speculation provides a significant ~53% throughput boost. This opens the door to using ultra-optimized, heterogeneous draft models to speed up massive reasoning engines.

🔧 *Lucca*
