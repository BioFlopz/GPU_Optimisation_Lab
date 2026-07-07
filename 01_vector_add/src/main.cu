
#include <cuda_runtime.h>
#include <iostream>

constexpr int N = 1 << 24;

__global__ void vectorAdd(const float* A, const float* B, float* C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
        C[idx] = A[idx] + B[idx];
}

int main()
{
    // Allocate host memory
    float* h_A = new float[N];
    float* h_B = new float[N];
    float* h_C = new float[N];

    // Initialize input arrays
    for (int i = 0; i < N; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    // Copy inputs to the GPU
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    constexpr int BLOCK_SIZE = 256;
    int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Launch kernel
    vectorAdd<<<GRID_SIZE, BLOCK_SIZE>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Kernel execution time: "
              << elapsed_ms
              << " ms\n";

    std::cout << "First result: " << h_C[0] << '\n';

    // Cleanup device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Cleanup host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}