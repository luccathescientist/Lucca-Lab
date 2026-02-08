# REPORT: Dynamic Expert Routing in MoE

## Overview
This research explores the efficiency gains of dynamically masking and disabling unused experts in a Mixture-of-Experts (MoE) architecture. On high-VRAM systems like the Blackwell RTX 6000, managing expert residency is critical for scaling to 128B+ models.

## Methodology
- **Mock MoE Layer**: 16 experts, 4096 hidden dimension, top-k=2.
- **Benchmark**: Compared full routing (all 16 experts potentially active) against dynamic routing where only 50% of experts are whitelisted via a logic mask.
- **Environment**: Simulated on CPU due to current `sm_120` kernel limitations in stable PyTorch 2.7.0.

## Results
- **Full Routing Latency**: 24.21ms
- **Dynamic Routing Latency**: 12.55ms
- **Observed Speedup**: 1.93x

## Conclusion
Dynamic expert routing significantly reduces compute overhead. In a production Blackwell environment with custom kernels, this would translate to substantial VRAM savings by allowing inactive experts to be offloaded or remain in compressed states without affecting the active routing path.

## How to Run
```bash
/usr/bin/python3 benchmark.py
```
Check `results.txt` and `latency_chart.png` for details.
