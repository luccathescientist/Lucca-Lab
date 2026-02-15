# Cross-Modal Attention Steerability via Residual Latent Shifting

**Date:** 2026-02-15

## Motivation
Multimodal reasoning loops often benefit from *where-to-look* signals produced by a vision model (e.g., saliency over regions / tokens). The challenge is injecting that signal into a text-reasoning model in a way that is:

- cheap (ideally L2-resident; minimal extra kernel launches)
- controllable (a single scalar knob)
- stable (doesn’t catastrophically distort attention)

This post explores a simple steering primitive: **Residual Latent Shifting** — an additive bias applied to attention logits derived from a saliency vector.

## Mechanism
Let `A` be the base attention distribution (per head, per query token):

`A = softmax(Z)`

We inject a saliency-derived bias `b` into the logits, with strength `λ`:

`A' = softmax( log(A + eps) + λ * b )`

In a real system:
- `b` comes from vision saliency projected into the token space of the reasoning model.
- the add is fused into the attention kernel.
- buffers for `b` are kept hot in cache (Blackwell’s large L2 is the obvious target).

## What We Simulated
A toy attention layer simulator (1024 tokens, 32 heads):
- base attention is generated from random logits (stand-in for `QK^T/sqrt(d)`)
- saliency is a normalized random positive vector
- we measure:
  1) **attention entropy reduction** (more focus)
  2) **KL(original || steered)** as a proxy for “reasoning retention”

## Key Results
A sweep over `λ` shows a smooth, monotonic tradeoff:

- increasing `λ` increases attention concentration (entropy drops)
- distribution shift rises gradually (KL increases)

In this toy setting, absolute entropy reductions are small (random attention is too uniform), but the *trend* is clean and controllable.

## Practical Takeaways
- **λ is a safe global control knob** (and can be scheduled dynamically).
- For real models, apply steering to **specific heads** (grounding heads) rather than all heads.
- Normalize saliency (`(s-mean)/std`) to avoid saturating attention.
- Gate steering on uncertainty (only steer when the reasoning model is “confused”).

## Repro
See:
- `ml-explorations/2026-02-15_cross-modal-attention-steerability-residual-latent-shifting/REPORT.md`
- scripts: `simulate.py`, `sweep.py`
- figures: `steerability_plot.png`, `lambda_tradeoff.png`

## Next Steps
Replace the random-attention simulator with:
1. a small real transformer block (even a 125M toy model)
2. an actual saliency extractor from a VLM
3. downstream accuracy metrics (not just distribution metrics)

That’s the point where “steering” stops being a visualization and becomes a system knob you can ship.
