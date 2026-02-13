# REPORT: Asynchronous Weight-Gradient Pipelining (AWGP) for Multi-Node Blackwell

## Overview
This research explores **Asynchronous Weight-Gradient Pipelining (AWGP)**, a specialized training strategy for the RTX 6000 Blackwell (sm_120) architecture. AWGP leverages Blackwell's advanced stream management and NVLink-7 bandwidth to overlap weight updates and gradient synchronizations with the subsequent forward pass.

## Methodology
The simulation modeled a multi-node Blackwell setup with the following parameters:
- **NVLink-7 Bandwidth**: 1.8 TB/s (simulated)
- **Tensor Core Throughput**: 1.8 PFLOPS FP8 (simulated)
- **Parameter Slice**: 10GB per node
- **Compute Latency**: 50ms per forward pass

We compared a **Sequential** baseline (compute -> communicate -> update) against the **AWGP** approach (compute || communicate).

## Results
- **Sequential Step Latency**: 55.56 ms
- **AWGP Step Latency**: 50.00 ms
- **Theoretical Speedup**: **1.11x**

By overlapping 100% of the communication overhead with the compute phase, AWGP effectively reduces the "idle" time of the Tensor Cores to near zero, provided the communication time does not exceed the compute time.

## Visualizations
The performance delta is captured in `awgp_performance.png`.

## How to Run
1. Ensure `matplotlib` is installed.
2. Execute `python3 simulate_awgp.py`.
3. Results will be logged to `results.txt` and a chart will be generated.
