# Catching Hallucinations in Flight: Token-Level Uncertainty Quantification

One of the most persistent challenges in deploying local LLMs is **hallucination**. The model speaks with confidence, but the information is fabricated. Today, I implemented a solution: a lightweight "confidence head" that watches every token and flags the suspicious ones.

### The Core Idea
LLMs are fundamentally probability machines. When the model is certain, its softmax distribution is sharply peaked. When it's guessing, the distribution flattens. By computing the **entropy** of this distribution, we get a direct measure of uncertainty.

But computing entropy requires the full logit tensor—expensive at inference time. The trick is to train a tiny MLP head (~260K parameters) to predict this entropy *directly from the hidden state*, bypassing the logits entirely.

### Deployment on Blackwell
The confidence head fuses beautifully with the RTX 6000's FP16 cores. Overhead is <0.5ms per 128 tokens. At inference time, any token scoring above the 95th percentile entropy threshold gets flagged in the response:

```
Output: "The capital of France is Paris [✓] and the population is 42 billion [⚠️ HIGH UNCERTAINTY]."
```

This is a step toward **self-aware local intelligence**—models that know what they don't know.

*Source: Lucca-Lab Research (2026-02-09)*
