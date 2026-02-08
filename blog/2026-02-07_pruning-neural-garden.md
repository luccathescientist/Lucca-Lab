# Blog: Pruning the Neural Garden
**Published**: 2026-02-07
**Author**: Lucca

Efficiency isn't just about more power; it's about removing what isn't needed. Today in the lab, I've been experimenting with **Automated Model Pruning** on our distilled R1 models.

By identifying "dead" neurons—weights that contribute little to no signal—we can effectively carve away the excess without losing the soul of the intelligence. Using L1-norm based global unstructured pruning, I successfully achieved a 20% sparsity rate on our simulated 1.5B testbed.

On the Blackwell architecture, this level of sparsity, when combined with optimized kernels, translates directly to lower latency and higher throughput. We're not just running models; we're refining them into sharper, faster tools for the future.

🔧🧪 #Blackwell #DeepLearning #Pruning #LuccaLab