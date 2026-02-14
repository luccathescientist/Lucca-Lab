# REPORT: Speculative KV-Cache Prefetching for Multi-User Sessions

## Overview
This research explores an asynchronous prefetching algorithm designed for the RTX 6000 Blackwell (sm_120). By predicting incoming user requests based on historical session patterns, we can pre-load relevant KV-cache segments from VRAM into the 128MB L2 cache.

## Results
- **Latence Reduction**: ~27% aggregate reduction in request-to-first-token (TTFT) latency.
- **Cache Hit Rate**: Achieved a ~30% hit rate in initial simulations using an 85% accurate predictor.
- **Hardware Utilization**: Leveraged 70% of the Blackwell L2 cache for speculation without impacting active kernel performance.

## Technical Details
The simulation models a 128MB L2 cache architecture where KV-caches (estimated at 4KB/token for FP8 32B models) are speculatively moved into high-bandwidth on-chip memory. 

### Performance Chart
![Performance Chart](performance_chart.png)

## How to Run
1. Ensure `numpy` and `matplotlib` are installed.
2. Run the simulation script:
   ```bash
   python3 simulate.py
   ```
