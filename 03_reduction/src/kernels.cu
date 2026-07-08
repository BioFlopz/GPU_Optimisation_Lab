#include "kernels.cuh"
#include "common.h"


__inline__ __device__
float warpReduce(float value)
{
    value += __shfl_down_sync(0xffffffff, value, 16);
    value += __shfl_down_sync(0xffffffff, value, 8);
    value += __shfl_down_sync(0xffffffff, value, 4);
    value += __shfl_down_sync(0xffffffff, value, 2);
    value += __shfl_down_sync(0xffffffff, value, 1);

    return value;
}


__global__ void reduceNaive(const float* input, float* output, int n)
{
    __shared__ float s_block[BLOCK_SIZE];

    unsigned int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIndex < n)
        s_block[threadIdx.x] = input[globalIndex];
    else
        s_block[threadIdx.x] = 0.0f;

    __syncthreads();


    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        unsigned int index = 2 * stride * threadIdx.x;

        if (index < blockDim.x)
        {
            s_block[index] += s_block[index + stride];
        }

        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        output[blockIdx.x] = s_block[0];
    }
}


__global__ void reduce_interleaved(const float* input, float* output, int n)
{
    __shared__ float s_block[BLOCK_SIZE];

    unsigned int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIndex < n)
        s_block[threadIdx.x] = input[globalIndex];
    else
        s_block[threadIdx.x] = 0.0f;

    __syncthreads();


    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            s_block[threadIdx.x] += s_block[threadIdx.x + stride];
        }

        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        output[blockIdx.x] = s_block[0];
    }
}




__global__ void reduce_v3(const float* input, float* output, int n)
{
    __shared__ float s_block[BLOCK_SIZE];

    unsigned int globalIndex = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float sum = 0.0f;

    if (globalIndex < n)
        sum += input[globalIndex];

    if (globalIndex + blockDim.x < n)
        sum += input[globalIndex + blockDim.x];

    s_block[threadIdx.x] = sum;

    __syncthreads();


    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            s_block[threadIdx.x] += s_block[threadIdx.x + stride];
        }

        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        output[blockIdx.x] = s_block[0];
    }
}



__global__ void reduce_v4(const float* input, float* output, int n)
{
    __shared__ float s_block[BLOCK_SIZE];

    unsigned int globalIndex = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float sum = 0.0f;

    if (globalIndex < n)
        sum += input[globalIndex];

    if (globalIndex + blockDim.x < n)
        sum += input[globalIndex + blockDim.x];

    s_block[threadIdx.x] = sum;

    __syncthreads();


    for (unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            s_block[threadIdx.x] += s_block[threadIdx.x + stride];
        }

        __syncthreads();
    }


    if (threadIdx.x < 32)
    {
        volatile float* vsmem = s_block;

        vsmem[threadIdx.x] += vsmem[threadIdx.x + 32];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 16];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 8];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 4];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 2];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 1];
    }


    if (threadIdx.x == 0)
    {
        output[blockIdx.x] = s_block[0];
    }
}




__global__ void reduce_v5(const float* input, float* output, int n)
{
    __shared__ float s_block[BLOCK_SIZE];

    unsigned int globalIndex = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float sum = 0.0f;

    if (globalIndex < n)
        sum += input[globalIndex];

    if (globalIndex + blockDim.x < n)
        sum += input[globalIndex + blockDim.x];

    s_block[threadIdx.x] = sum;

    __syncthreads();


    for (unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            s_block[threadIdx.x] += s_block[threadIdx.x + stride];
        }

        __syncthreads();
    }


    if (threadIdx.x < 32)
    {
        float value = s_block[threadIdx.x];

        if (threadIdx.x + 32 < blockDim.x)
            value += s_block[threadIdx.x + 32];

        value = warpReduce(value);

        if (threadIdx.x == 0)
            output[blockIdx.x] = value;
    }
}