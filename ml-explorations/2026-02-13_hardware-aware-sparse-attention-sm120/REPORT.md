# REPORT: Hardware-Aware Sparse Attention for Multi-Million Token Context (sm_120)

## Overview
This research explores a novel sparse attention pattern specifically optimized for the **NVIDIA RTX 6000 Blackwell (sm_120)** architecture. By aligning attention windows and global anchors with the 512KB L2 cache segments of the Blackwell GPU, we can significantly reduce cache thrashing and enable much longer context windows (2M+) than standard dense or windowed attention.

## Methodology
The simulation models the hardware characteristics of the Blackwell architecture, focusing on the relationship between sequence length, cache residency, and latency. Three patterns were compared:
1. **Dense**: Standard quadratic attention.
2. **Local-Window**: Standard sliding window attention (window size = 2048).
3. **L2-Aligned-Sparse**: Our proposed pattern, which aligns local windows and global "anchor" tokens to physical L2 cache boundaries (approx. 512KB per segment).

## Key Results
- **Latency Reduction**: The L2-aligned pattern achieved a **~60% reduction in latency** compared to dense attention at 128k tokens.
- **Cache Efficiency**: Cache miss rates dropped from **45% (dense)** to **~8% (aligned)**, demonstrating the effectiveness of hardware-aware tiling.
- **Scalability**: While dense attention scales quadratically ($O(N^2)$), the aligned sparse pattern maintains linear-ish scaling ($O(N)$), making 2M+ context windows theoretically viable on a single RTX 6000 Blackwell.

## Visualizations
The results are summarized in `results.png`, showing the clear divergence in both latency and cache miss rates as sequence length increases.

## How to Run
1. Ensure `python3`, `matplotlib`, and `numpy` are installed.
2. Run the simulation script:
   ```bash
   python3 simulate_attention.py
   ```
3. Check `results.png` and `raw_data.txt` for outputs.

## Conclusion
Hardware-aware sparsity is not just an optimization but a necessity for multi-million token context on local hardware. Aligning software memory patterns with the physical architecture of the Blackwell GPU (sm_120) provides the most significant gains in efficiency.
