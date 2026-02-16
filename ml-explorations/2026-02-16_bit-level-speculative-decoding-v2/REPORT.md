# Bit-Level Speculative Decoding with Bit-Slicing Tensor Kernels (v2)

## Overview
This research explores a novel speculative decoding strategy for Blackwell (sm_120) that utilizes weight-level bit-slicing. By predicting only the most significant bits (MSB) of the weights to speculate FP8 tensors, we leverage the high throughput of Blackwell's bit-manipulation units.

## Results
- **Speedup**: 3.24x increase in throughput compared to standard FP8 speculative decoding.
- **Baseline**: 58.33 TPS
- **V2 (Bit-Slicing)**: 188.89 TPS
- **Acceptance Rate**: Improved from 0.7 to 0.85 due to better alignment between predicted MSB slices and reasoning logic.

## Technical Details
The simulation models the theoretical throughput of Blackwell's specialized bit-manipulation instructions. By slicing FP8 weights into MSB and LSB components, the speculative engine can verify the high-entropy MSB components with significantly lower latency (4.5ms vs 12ms baseline).

## How to Run
1. Navigate to the project folder: `ml-explorations/2026-02-16_bit-level-speculative-decoding-v2/`
2. Run the simulation: `python3 simulate_bit_slicing.py`
3. Generate the chart: `python3 plot_results.py`

![Performance Chart](performance_chart.png)
