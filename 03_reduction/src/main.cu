#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cstdlib>

#include "cuda_check.h"
#include "kernels.cuh"
#include "common.h"


inline float cpuReduce(const float* data, int size)
{
    float sum = 0.0f;

    for (int i = 0; i < size; i++)
        sum += data[i];

    return sum;
}


int main()
{
    const size_t bytes = N * sizeof(float);

    float* h_input = new float[N];

    for (int i = 0; i < N; i++)
        h_input[i] = 1.0f;

    float cpuResult = cpuReduce(h_input, N);

    float* d_input;
    float* d_output;

    // int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blocks = (N + (BLOCK_SIZE * 2) - 1) / (BLOCK_SIZE * 2);

    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, blocks * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    reduce_v4<<<blocks, BLOCK_SIZE>>>(d_input, d_output, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;

    cudaEventElapsedTime(&ms, start, stop);

    float* h_partial = new float[blocks];

    CHECK_CUDA(cudaMemcpy(h_partial, d_output, blocks * sizeof(float), cudaMemcpyDeviceToHost));

    float gpuResult = cpuReduce(h_partial, blocks);

    std::cout << "CPU Result : " << cpuResult << '\n';

    std::cout << "GPU Result : " << gpuResult << '\n';

    std::cout << "Kernel Time : " << ms << " ms\n";

    delete[] h_input;
    delete[] h_partial;

    cudaFree(d_input);
    cudaFree(d_output);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}