# Bit-Level Speculative Decoding with Tensor Slicing

## Overview
This research explores a novel speculative decoding strategy for the Blackwell architecture (sm_120). By predicting only the higher-order bits (MSB) of tensors using a lightweight 1B student model, we can verify tokens using a "bit-sliced" version of the target model (R1-70B).

## Results
- **Speculator Overhead**: 1.5ms
- **Full FP8 Latency**: 50.00ms
- **Sliced Verification Latency**: 29.41ms
- **Total Speculative Step**: 30.91ms
- **Projected Speedup**: 1.62x

## Technical Insights
The primary speedup comes from the Blackwell architecture's ability to handle sub-INT8 operations with significantly higher throughput. By slicing FP8 tensors into high-precision and low-precision components, the verification engine only needs to process the "critical" bits to confirm a speculator's guess.

## How to Run
1. Ensure Python 3.x is installed.
2. Run `python3 simulate_speculation.py` to view the latency profiles.
3. Run `python3 generate_charts.py` to regenerate result visualizations.
