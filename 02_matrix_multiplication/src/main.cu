#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cstdlib>

#include "cuda_check.h"
#include "kernels.cuh"
#include "constants.h"




int main()
{
    size_t bytes = N * N * sizeof(float);

    float* h_A = new float[N * N];
    float* h_B = new float[N * N];
    float* h_C = new float[N * N];

    for (int i = 0; i < N * N; i++)
    {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float* d_A;
    float* d_B;
    float* d_C;

    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matrixMulKernel_shared_unrolled<<<grid, block>>>(d_A, d_B, d_C, N);

    CHECK_CUDA(cudaGetLastError());

    cudaEventRecord(stop);

    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;

    cudaEventElapsedTime(&ms, start, stop);

    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    std::cout << "Kernel Time: " << ms << " ms\n";

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}