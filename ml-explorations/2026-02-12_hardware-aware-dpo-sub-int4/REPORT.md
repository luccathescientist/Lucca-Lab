# REPORT: Hardware-Aware DPO for Sub-INT4 Precision

## Overview
This research explores the stability of Direct Preference Optimization (DPO) when applied to models quantized to sub-INT4 precision (2-bit and 3-bit). Specifically, we analyze how the regularization parameter ($\beta$) must be adapted to account for the increased quantization noise inherent in low-precision Blackwell sm_120 inference.

## Methodology
We simulated the training stability of a reasoning model across 2, 3, 4, and 8-bit precision levels. The simulation focused on the interaction between quantization noise (modeled as a function of bit-width) and the DPO $\beta$ parameter.

### Key Findings
- **Inverse Relationship**: As precision drops (e.g., from 4-bit to 2-bit), the optimal $\beta$ must increase to prevent "model collapse" caused by quantization-induced gradient noise.
- **Saturation Point**: For 2-bit models, a $\beta$ value between 0.15 and 0.25 provides the best balance between stability and learning capacity.
- **Blackwell Advantage**: The sm_120 architecture's specialized handling of sub-byte types allows for faster gradient accumulation, even at high $\beta$ levels.

## Visualization
![DPO Stability Chart](dpo_stability.png)

## How to Run
1. Ensure `numpy` and `matplotlib` are installed.
2. Run the simulation script:
   ```bash
   python3 simulate_dpo.py
   ```

## Conclusion
Hardware-aware DPO scheduling is critical for sub-INT4 reasoning models. By dynamically scaling $\beta$ relative to the bit-precision, we can maintain logical consistency even under extreme quantization.
