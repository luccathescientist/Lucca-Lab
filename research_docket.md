# Laboratory Research Docket

## High-Level Goals
- Optimize local inference for Blackwell architecture.
- Expand multi-modal integration (Vision + Motion).
- Automate scientific documentation and data archival.

## Pending Tasks
- [x] **Autonomous Rig Self-Optimization**: Research dynamic VRAM allocation strategies for heterogeneous model pools (R1, Flux, Wan 2.1) to eliminate swapping overhead.
- [x] **Quantized Mixture-of-Experts (Q-MoE)**: Evaluate the feasibility of sub-4-bit MoE quantization on Blackwell Tensor Cores without significant routing collapse.
- [x] **Temporal Consistency in I2V**: Develop a "Memory-Aware Motion" pipeline using Wan 2.1 and a temporal LoRA to maintain character identity across multiple video clips.
- [x] **Neural Knowledge Distillation for C++**: Distill specialized CUDA/C++ systems programming expertise from o3-mini/R1 into a small 1B student model.
- [x] **Federated Learning for Local Intelligence**: Prototype a secure, local-first federated learning node to share model weights across multiple household rigs (if detected).
- [x] **Autonomous Documentation Generator**: Build an agent that watches `git commit` history and auto-generates Markdown documentation for new lab features.
- [x] **FlashAttention-3 Integration**: Benchmark FP8 kernels on Blackwell RTX 6000 for 2x throughput gains.
- [x] **Video-to-Text-to-Action (VTA)**: Chain Wan 2.1 video analysis with R1 to generate autonomous robotic control scripts.
- [x] **Model Merging (Evolutionary Algorithm)**: Use R1 to guide the merging of specialized Llama-3 fine-tunes for a "Super-Generalist" local model.
- [x] **On-Device DPO (Direct Preference Optimization)**: Implement a lightweight DPO pipeline to fine-tune R1-1.5B based on the Lead Scientist's feedback loop.
- [x] **Cross-Modal Retrieval Augmented Generation (CM-RAG)**: Index images and videos from the Lab Dashboard into a vector DB for visual recall.
- [x] **Neural Architecture Search (NAS) for Blackwell**: Identified critical kernel desync for `sm_120`; validated theoretical GQA advantages.
- [x] **Cross-Model Knowledge Distillation (DeepSeek-R1-Teacher -> local R1-32B)**: Evaluate logit-matching efficiency on Blackwell for specialized logic domains.
- [x] **Sparse Attention mechanisms for long-context R1 stability**: Implementation of Block-Sparse Attention to maintain 128k context without OOM.
- [x] **Automated LoRA switching for multi-personality task handling**: Build a router that swaps Flux LoRAs based on prompt sentiment analysis.
- [x] **Benchmark INT8-Quantized KV Cache vs FP8 on Blackwell**: Determine throughput/perplexity tradeoffs on Compute 12.0.
- [x] **Neural Heatmap Visualization**: Generate real-time 3D attention maps of R1 reasoning paths.
- [x] **Quantum-Resistant Model Merging**: Investigate the impact of merging weights on encryption-standard-level logic (NIST benchmarks) using FP8 precision.
- [x] **Dynamic KV-Cache Pruning**: Implement a real-time attention-mask-based KV cache pruner to extend effective context window beyond 128k on Blackwell.
- [ ] **Neural Feedback Loop (Reflexion v2)**: Build a recursive self-correction pipeline where R1 reviews its own CUDA kernels and optimizes for sm_120 register pressure.
- [ ] **Visual-Temporal State Tracking**: Chain Wan 2.1 motion vectors with Qwen2-VL to track object state changes (e.g., "cup filled" vs "cup empty") in long-form video.
- [ ] **FP8 Tensor-Core Bit-Slicing**: Research experimental bit-slicing techniques to squeeze sub-INT4 performance out of Blackwell's FP8 tensor cores.
- [ ] **Autonomous RAG Synthesis**: Use R1-32B to autonomously synthesize and de-duplicate the entire `Lucca-Lab` repository into a high-density "Scientific Knowledge Graph".

## Completed Tasks
- [x] Create Spatial Reasoning Loop (Qwen2-VL frames -> DeepSeek-R1 analysis).
- [x] Conduct Context Window Stress Test (8k to 32k tokens) with FP8 KV cache.
- [x] Implement Proactive Discord Alerts (Send 8-hour summaries to #tech automatically).
- [x] Test the email transmission pipeline within the autonomous scientist loop.
- [x] Benchmark DeepSeek-V3 vs R1 on Blackwell inference latency.
- [x] Implement Model Chaining: Llama-3-Vision -> OCR -> R1 Reasoning.
- [x] Evaluate 4-bit vs 8-bit quantization quality loss on math benchmarks.
- [x] Optimize vLLM PagedAttention for concurrent multi-agent requests.
- [x] Implement "Neural Dreaming" (automatic generation of synthetic training data for small models).
- [x] Test cross-model speculative decoding (different architectures).
- [x] Auto-generate unit tests for local ML scripts using GPT-5.2 Codex.
- [x] Stress test 96GB VRAM limits with concurrent Flux and Wan 2.1 runs.
- [x] Evaluate OpenAI o3-mini vs local R1-32B on engineering logic tasks.
- [x] Implement Automated Model Pruning: Remove dead neurons from smaller Distilled models.
- [x] Neural Interface 3D Hologram: Research Three.js integration for real-time rig visualization.
- [x] Implement Speculative Decoding via FP8 (Draft: R1-1.5B, Target: R1-8B).
- [x] Implement Model Laboratory in dashboard (choose model + prompt + streaming).
- [x] Implement Native FP8 Acceleration for R1-70B (Initialized on 8001).
- [x] Implement Speculative Decoding for R1-70B (Evaluating FP8 draft models).
- [x] Deploy Blackwell-optimized vLLM FP8 kernels for DeepSeek-R1-32B.
