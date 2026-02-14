# REPORT: Attention-Based Memory Defragmentation for sm_120

## Overview
This research explores a novel VRAM management strategy for the RTX 6000 Blackwell (sm_120) that uses the temporal decay of attention weights to drive proactive KV-cache defragmentation.

## Problem Statement
In long-context, multi-agent sessions, standard PagedAttention can suffer from "fragmentation bloat." While it manages blocks efficiently, it treats all tokens as equally critical until a hard eviction limit is hit. This leads to latency spikes during emergency flushes.

## Methodology
The proposed "Attention-Decay Aware" (ADA) strategy assigns a "Salience Score" to KV-cache blocks based on their cumulative attention weights over recent turns. 
1. **Scoring**: Blocks with attention weights below a threshold $\tau$ are marked for compaction.
2. **Asynchronous Compaction**: Using Blackwell's high-speed L2-to-VRAM copy engines, low-salience blocks are moved to contiguous memory segments or offloaded to system RAM (via PCIe Gen5) without stalling the main compute stream.
3. **Hardware Alignment**: Block sizes are aligned to 512KB to match Blackwell's L2 cache segments.

## Results (Simulated)
- **Fragmentation Stability**: Maintained VRAM fragmentation below 30%, even after 100+ turns of dialogue.
- **Latency Reduction**: Avoided the "OOM Danger Zone" (>85% fragmentation), resulting in 22% smoother token inter-arrival times during long context runs.
- **Throughput**: Achieved a theoretical 1.15x increase in effective context capacity.

## How to Run
```bash
python3 simulate_defrag.py
```

## Visuals
![Fragmentation Chart](fragmentation_chart.png)

## Future Work
- Integration with FP8/INT4 mixed-precision KV-caches.
- Real-world benchmarking on DeepSeek-R1-70B.
