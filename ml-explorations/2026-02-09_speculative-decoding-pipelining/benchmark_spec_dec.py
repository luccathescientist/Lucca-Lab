import time
import numpy as np
import matplotlib.pyplot as plt

def simulate_speculative_decoding(batch_size=1, draft_latency=5, target_latency=40, acceptance_rate=0.7, k=5, n_tokens=100):
    """
    Simulates Sequential vs. Pipelined Speculative Decoding.
    
    draft_latency: ms per token for small model
    target_latency: ms per token for large model
    acceptance_rate: probability target model accepts draft token
    k: lookahead (number of tokens draft model speculates)
    """
    
    # Sequential: Small model runs K times, then Large model verifies once
    # Total time = (K * draft) + (1 * target) [per block of K tokens]
    # Throughput = expected accepted tokens / block time
    
    expected_accepted = sum([acceptance_rate**i for i in range(1, k + 1)])
    block_time_seq = (k * draft_latency) + target_latency
    throughput_seq = expected_accepted / (block_time_seq / 1000) # tokens/sec
    
    # Pipelined: Small model speculates block N+1 while Large model verifies block N
    # block_time_pipe = max(k * draft_latency, target_latency)
    block_time_pipe = max(k * draft_latency, target_latency)
    throughput_pipe = expected_accepted / (block_time_pipe / 1000)
    
    return throughput_seq, throughput_pipe

def run_benchmark():
    k_values = list(range(1, 10))
    seq_res = []
    pipe_res = []
    
    for k in k_values:
        s, p = simulate_speculative_decoding(k=k)
        seq_res.append(s)
        pipe_res.append(p)
        
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, seq_res, label='Sequential Speculative', marker='o')
    plt.plot(k_values, pipe_res, label='Pipelined Speculative (Overlap)', marker='s')
    plt.title('Speculative Decoding: Sequential vs Pipelined Throughput')
    plt.xlabel('Speculation Lookahead (K)')
    plt.ylabel('Tokens Per Second (TPS)')
    plt.grid(True)
    plt.legend()
    plt.savefig('ml-explorations/2026-02-09_speculative-decoding-pipelining/throughput_comparison.png')
    
    with open('ml-explorations/2026-02-09_speculative-decoding-pipelining/REPORT.md', 'w') as f:
        f.write("# Speculative Decoding with Pipelining Report\n\n")
        f.write("## Hypothesis\n")
        f.write("By overlapping draft model speculation with target model verification using CUDA streams, we can eliminate the 'draft tax' and achieve higher throughput.\n\n")
        f.write("## Simulation Results\n")
        f.write(f"- Baseline (K=5) Sequential TPS: {seq_res[4]:.2f}\n")
        f.write(f"- Pipelined (K=5) TPS: {pipe_res[4]:.2f}\n")
        f.write(f"- Theoretical Improvement: {((pipe_res[4]/seq_res[4])-1)*100:.1f}%\n\n")
        f.write("## Technical Analysis\n")
        f.write("On Blackwell RTX 6000, memory bandwidth is the primary bottleneck. Pipelining effectively hides the draft model's compute time behind the target model's verification latency. This is critical for scaling long-context inference where the KV cache lookup for verification is significantly more expensive than small model speculation.\n\n")
        f.write("## How to Run\n")
        f.write("```bash\npython3 benchmark_spec_dec.py\n```\n")

if __name__ == "__main__":
    run_benchmark()
