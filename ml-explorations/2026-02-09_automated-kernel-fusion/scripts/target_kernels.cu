#include <cuda_runtime.h>
#include <iostream>

// Kernel 1: Element-wise addition
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// Kernel 2: Element-wise multiplication
__global__ void vectorMul(const float* A, const float* B, float* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] * B[i];
    }
}

// Kernel 3: ReLU activation
__global__ void vectorReLU(float* A, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        A[i] = fmaxf(0.0f, A[i]);
    }
}

int main() {
    int N = 1024 * 1024;
    size_t size = N * sizeof(float);

    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch separate kernels
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    vectorMul<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_C, N);
    vectorReLU<<<blocksPerGrid, threadsPerBlock>>>(d_C, N);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
