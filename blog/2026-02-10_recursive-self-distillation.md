# Recursive Self-Distillation: Squeezing Reasoning into Smaller Models

Distilling a large model into a small one is common, but *recursive self-distillation* adds a critical filtering layer. By using R1-70B to verify and refine its own reasoning chains before they ever reach the student, we ensure that the student learns only the "golden path" of logic.

In our latest lab run on the Blackwell RTX 6000, we simulated this pipeline. We found that while raw teacher data is plentiful, the "Refined Essence" leads to more stable training and better logical consistency. This is a primary path for our goal of achieving 70B-level reasoning on 32B-parameter local rigs.

![Loss Convergence Chart](../../ml-explorations/2026-02-10_recursive-self-distillation/distillation_plot.png)
