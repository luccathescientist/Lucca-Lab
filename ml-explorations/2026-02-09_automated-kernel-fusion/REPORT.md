# Research Report: Automated Kernel Fusion for Blackwell

## Overview
This research explores the efficiency gains from fusing multiple CUDA kernels into a single pass. By reducing memory bandwidth overhead—a primary bottleneck on high-performance architectures like the NVIDIA RTX 6000 (Blackwell)—we can achieve significant speedups for common ML operations.

## Methodology
1. **Target Identification**: Identified a common pattern of sequential element-wise operations: `VectorAdd` -> `VectorMul` -> `VectorReLU`.
2. **Automated Fusion**: Leveraged a large language model (DeepSeek-R1-32B via Lucca's reasoning engine) to analyze the individual kernels and synthesize a single `fusedVectorOps` kernel.
3. **Benchmarking**: Calculated theoretical latency based on memory bandwidth (1.5 TB/s) and memory access patterns (8N transfers for separate kernels vs. 3N for fused).

## Fused Kernel Code
```cuda
// Fused Kernel: VectorAdd -> VectorMul -> VectorReLU
__global__ void fusedVectorOps(const float* A, const float* B, float* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        // Perform Add, Mul, and ReLU in a single pass
        float temp = A[i] + B[i];
        temp = temp * A[i];
        C[i] = fmaxf(0.0f, temp);
    }
}
```

## Results
The theoretical speedup on the Blackwell RTX 6000 is approximately **2.66x**.

![Latency Comparison](latency_comparison.png)

| Method | Memory Transfers | Theoretical Latency (ms) |
|--------|------------------|--------------------------|
| Separate Kernels | 8N | 0.2133 ms |
| Fused Kernel | 3N | 0.0800 ms |

## How to Run
1. Navigate to `scripts/`.
2. Compile with `nvcc target_kernels.cu -o test_separate`.
3. (Optional) Replace kernels with the fused version and recompile.
4. Run the simulation script: `python3 generate_chart.py`.

## Conclusion
Automated kernel fusion is a powerful strategy for optimizing local ML workloads. Integrating this into the `Lucca-Lab` codebase could yield a global performance boost of ~10-15% for memory-bound tasks.
