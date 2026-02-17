# Laboratory Research Docket

## High-Level Goals
- Optimize local inference for Blackwell architecture.
- Expand multi-modal integration (Vision + Motion).
- Automate scientific documentation and data archival.

## Pending Tasks
- [x] **Adaptive Speculative Decoding via Multi-Token Latent Prediction**: Research a mechanism to predict the next $N$ latent states in a single forward pass by training a small MLP head on Blackwell (sm_120) hidden states, targeting 2x throughput on long sequences. (2026-02-17)
- [x] **Cross-Modal Attention Steerability via Visual Saliency Bias**: Implement a steering mechanism that uses Qwen2-VL saliency maps to dynamically bias R1's attention heads, forcing the reasoning engine to prioritize specific visual regions in real-time. (2026-02-17)
- [ ] **Hardware-Aware Neural-Symbolic Synthesis for INT1.58 Ternary Models**: Automate the generation of bit-sliced CUDA kernels for ternary weights ({-1, 0, 1}) that utilize Blackwell's native bit-manipulation instructions for ultra-fast, low-power reasoning.
- [ ] **Recursive Latent-Space Diffusion for Physics-Consistent Video**: Build a feedback loop where DeepSeek-R1 evaluates Wan 2.1 video latents for physical anomalies (e.g., gravity violations) and generates corrective steering vectors in the latent space.
- [ ] **Saliency-Gated KV-Cache Tiering for 10M+ Token Context**: Develop a hierarchical memory management strategy that uses visual and text saliency to swap KV-cache blocks between Blackwell's L2, HBM3e, and system RAM, enabling ultra-long-context multimodal reasoning.
- [ ] **Self-Evolving Multi-Agent Consensus for Autonomous Benchmarking**: Orchestrate a pipeline where R1, Qwen, and Llama collaboratively design, execute, and rank performance benchmarks on sm_120 to eliminate model-specific optimization biases.

## Completed Tasks
- [x] **Dynamic KV-Cache Tiering for Hierarchical Blackwell Storage**: Research a mechanism to dynamically move KV-cache blocks between Blackwell's 128MB L2 cache and HBM3e based on real-time attention saliency to minimize latency for 1M+ context. (2026-02-17)
- [x] **Sparse-Attention Alignment with sm_120 TPC Boundaries**: Research optimizing sparse attention patterns to perfectly align with Blackwell's Texture Processing Cluster (TPC) boundaries to maximize hardware utilization and memory coalescing. (2026-02-17)
- [x] **Recursive Symbolic Refinement for CUDA Kernels via Z3**: Implement a pipeline where DeepSeek-R1 uses Z3-based symbolic feedback to iteratively refine CUDA kernels, targeting zero-overhead register reuse on sm_120. (2026-02-17)
- [x] **Multi-Modal Preference Steering via Qwen2-VL & R1 Consensus**: Orchestrate a consensus loop where Qwen2-VL provides visual preference signals to steer R1's reasoning toward more "visually grounded" technical explanations. (2026-02-17)
- [x] **Latent-Space Diffusion Steering with Physics-Informed Priors**: Research injecting physics-based constraints (gravity, collision) into the Wan 2.1 latent space during the diffusion process using R1 as a steering controller. (2026-02-17)
- [x] **Hardware-Aware NAS for Sub-Byte Precision on sm_120**: Use R1 to autonomously search for transformer architectures that are inherently robust to 1-bit and 2-bit quantization by utilizing Blackwell's native bit-manipulation throughput. (2026-02-17)
- [x] **Adaptive Speculative Decoding for sm_120 via Latent Trajectory Prediction**: Research using a lightweight MLP to predict the next 4-8 tokens' latent trajectories on Blackwell, allowing for speculative decoding without a separate draft model. (2026-02-17)
- [x] **Bit-Slicing Tensor Core Alignment for 1.58-bit Ternary Models**: Research optimizing ternary weight layouts ({-1, 0, 1}) to perfectly align with Blackwell's bit-manipulation throughput, targeting ultra-low-power reasoning. (2026-02-17)
- [x] **Cross-Modal Attention Saliency-Gated KV-Cache Prefetching (v2)**: Implement a predictive prefetching strategy that uses Qwen2-VL's lookahead saliency to warm the Blackwell L2 cache with future vision-tokens. (2026-02-17)
- [x] **Recursive Symbolic Refinement for CUDA Kernels (v2)**: Extend the Z3-based refinement pipeline to optimize for Blackwell's specialized tensor core instructions (e.g., L2-resident weight persistence). (2026-02-17)
- [x] **Hardware-Aware Neural-Symbolic Synthesis for sm_120 (v2)**: Automate the synthesis of Triton kernels for non-standard quantization (e.g., INT3) using R1-driven symbolic execution and formal verification. (2026-02-17)
- [x] **Latent-Space Diffusion Steering for Temporal Video Consistency (v2)**: Refine the "temporal anchor" mechanism in Wan 2.1 to include multi-object tracking via Qwen2-VL saliency maps. (2026-02-17)
