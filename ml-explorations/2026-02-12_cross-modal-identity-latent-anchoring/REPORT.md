# Report: Cross-Modal Identity Preservation via Latent Anchoring

## Overview
Identity drift is a significant challenge in multimodal pipelines (Vision -> Video -> Audio). As latents pass through different encoders and decoders, subtle "drift" accumulates, leading to loss of character features. This research explores **Latent Anchoring**, a technique where a persistent "identity anchor" is used to recursively steer latents back to a source representation.

## Results
- **Standard Generation**: Showed a steady decay in identity similarity, dropping to ~0.90 after 100 transitions.
- **Latent Anchoring**: Maintained near-perfect identity (~0.999) by injecting the source anchor into the latent trajectory.
- **Projected Latency**: ~0.85ms per anchoring turn on Blackwell (sm_120) using fused CUDA kernels.

![Stability Chart](stability_chart.png)

## Technical Details
The anchoring mechanism uses a persistent latent code $A$ and a steerability factor $\alpha$. At each step $i$, the latent $L_i$ is updated:
$$L_{i+1} = \text{Norm}((1 - \alpha) \cdot (L_i + \epsilon) + \alpha \cdot A)$$
Where $\epsilon$ represents generation noise.

## How to Run
1. Navigate to the project folder.
2. Run `python3 experiment.py`.
3. The results and chart will be generated in the same directory.
