#include "kernels.cuh"
#include "common.h"

__global__ void reduceNaive(const float* input, float* output, int n)
{
    __shared__ float shared[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

    // Load into shared memory
    if (globalIndex < n)
        shared[tid] = input[globalIndex];
    else
        shared[tid] = 0.0f;

    __syncthreads();

    // Naive reduction
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0)
        {
            shared[tid] += shared[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0)
    {
        output[blockIdx.x] = shared[0];
    }
}