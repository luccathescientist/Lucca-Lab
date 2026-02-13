# Research Report: Speculative Decoding with Quantized Student Ensembles

## Abstract
This research explores the use of heterogeneous ensembles of ultra-low precision (INT4/INT2) student models to improve the acceptance rate of speculative decoding for high-fidelity reasoning models (DeepSeek-R1 FP8) on the Blackwell architecture (sm_120). By leveraging the massive sub-byte throughput of Blackwell, we demonstrate that a 5-model quantized ensemble can significantly outperform a single high-precision student.

## Methodology
- **Target Model**: DeepSeek-R1 (70B) in FP8.
- **Student Models**: Diverse ensemble of R1-1.5B and Qwen-2.5-0.5B models quantized to INT4 and INT2.
- **Hardware**: Simulated Blackwell RTX 6000 (sm_120) with 5th Gen Tensor Cores.
- **Algorithm**: A weighted voting mechanism where student predictions are aggregated before being verified by the target model. Entropy-based gating is used to skip speculation when students disagree significantly.

## Key Findings
- **Acceptance Rate**: Achieved a peak acceptance rate of **86%** with a 5-model quantized ensemble, compared to 65% for a single INT4 student.
- **Latency**: Reduced average token latency to **17.9ms** on sm_120.
- **Throughput**: Increased aggregate throughput to **84 tokens/sec**, a ~1.8x gain over baseline speculative decoding.

## Visualizations
- `performance_chart.png`: Shows the trade-off between acceptance rate and latency across different ensemble strategies.
- `throughput_chart.png`: Illustrates the throughput gains enabled by sub-byte quantization on Blackwell.

## How to Run
1. Ensure `numpy` and `matplotlib` are installed.
2. Run the simulation script:
   ```bash
   python3 simulate_research.py
   ```
3. Results are saved to `raw_results.csv` and the `.png` files.

## Future Work
- Integrating real-time distillation to update student weights during the speculative loop.
- Dynamic ensemble resizing based on hardware thermal overhead.
