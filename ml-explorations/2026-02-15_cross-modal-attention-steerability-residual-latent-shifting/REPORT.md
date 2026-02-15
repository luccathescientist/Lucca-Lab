# Cross-Modal Attention Steerability via Residual Latent Shifting (Simulation)

Date: 2026-02-15

## Goal
Explore a *lightweight* mechanism for steering a reasoning model’s attention distribution using a vision model’s saliency signal.

Concept: given a token-level saliency vector `s` (derived from Qwen2-VL / vision encoder), inject an additive residual bias into the attention logits of a reasoning model (R1) *before* softmax, ideally with the whole bias living in L2 (Blackwell 128MB L2) to keep the overhead sub-millisecond.

## Method (Toy Simulator)
We simulate one attention layer:

- Sequence length: **1024** tokens
- Heads: **32**
- Base attention: `A = softmax(Z)` where `Z ~ Normal(0, 0.1)` (stands in for `QK^T/sqrt(dk)`)
- Saliency: `s = normalize(|Normal(0,1)|)`

**Residual Latent Shifting** (steered attention):

`A' = softmax( log(A + eps) + λ * bias(s) )`

where `bias(s)` is broadcast across `[heads, q_tokens, k_tokens]`.

### Metrics
1. **Entropy reduction (%):** measures concentration (lower entropy = more focused attention)
2. **KL(original || steered):** distribution shift proxy (smaller = better “reasoning retention”)

## Results
Single-run sample (`λ = 0.25`):
- Entropy reduction: **~0.0093%**
- KL divergence: **~6.30e-4**
- Estimated throughput overhead (assumption, L2-resident bias add + logit add): **~0.85 ms**

### λ sweep (mean over 3 seeds)
The steerability/shift tradeoff is smooth and monotonic:

| λ | Entropy reduction (%) | KL(original||steered) |
|---:|---:|---:|
| 0.00 | -0.0000 | -0.000000 |
| 0.05 | 0.0005 | 0.000034 |
| 0.10 | 0.0020 | 0.000135 |
| 0.15 | 0.0044 | 0.000305 |
| 0.25 | 0.0125 | 0.000852 |
| 0.40 | 0.0326 | 0.002201 |
| 0.60 | 0.0751 | 0.005015 |
| 0.80 | 0.1368 | 0.009027 |
| 1.00 | 0.2192 | 0.014284 |

Plots:
- `steerability_plot.png` (token-level example)
- `lambda_tradeoff.png` (entropy vs KL tradeoff)

## Interpretation
- In this toy setup (random attention), the entropy reductions are *small* because the base attention is already close to uniform and the bias is weak relative to the full logit scale. In a real model, `QK^T` logits are structured and can be more sensitive to additive biases.
- The key operational handle is **λ**:
  - small λ: almost no distribution shift (low KL)
  - larger λ: increasing steerability but growing shift

## Implementation Sketch (Real System)
1. Run Qwen2-VL to produce a saliency/importance map over image regions.
2. Project saliency into token space (`s_k`) aligned to vision tokens (or cross-attended tokens).
3. Inject `λ * s_k` into the **key-token dimension** of selected attention heads/layers.
4. Keep saliency projection buffers in L2 (or persistent SRAM/shared) and update per-turn.

Possible refinements:
- Apply bias to a subset of heads designated “grounding heads”.
- Use a *temperature-scaled* bias: `λ * (s - mean(s)) / std(s)`.
- Gate injection only when R1’s self-reported uncertainty spikes.

## How to Run (Repro)
A local virtualenv is assumed.

```bash
# from repo root
python3 -m venv lab_venv
source lab_venv/bin/activate
pip install torch matplotlib numpy

# run single visualization
python3 ml-explorations/2026-02-15_cross-modal-attention-steerability-residual-latent-shifting/simulate.py

# run λ sweep (generates lambda_tradeoff.png)
cd ml-explorations/2026-02-15_cross-modal-attention-steerability-residual-latent-shifting
python3 sweep.py
```

## Files
- `simulate.py` — one-shot simulation + `steerability_plot.png`
- `sweep.py` — λ sweep + `lambda_tradeoff.png`

## Next Steps
- Replace random attention with a small real transformer attention block and measure *downstream task accuracy* impact.
- Benchmark overhead on GPU: implement bias add in a fused attention kernel (Triton) to keep it in-register/L2.
