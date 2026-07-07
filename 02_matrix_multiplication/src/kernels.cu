#include "kernels.cuh"
#include "constants.h"

__global__ void matrixMulKernel(const float* A, const float* B, float* C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N)
        return;

    float sum = 0.0f;

    for (int k = 0; k < N; k++)
    {
        sum += A[row * N + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}


__global__ void matrixMulKernel_shared(const float* A, const float* B, float* C, int N)
{
    __shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Number of tiles along one matrix dimension
    int numTiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int tile = 0; tile < numTiles; tile++)
    {
        // Global indices to load
        int tiledColA = tile * BLOCK_SIZE + threadIdx.x;
        int tiledRowB = tile * BLOCK_SIZE + threadIdx.y;

        // Load tile from A
        if (row < N && tiledColA < N)
            tileA[threadIdx.y][threadIdx.x] = A[row * N + tiledColA];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile from B
        if (tiledRowB < N && col < N)
            tileB[threadIdx.y][threadIdx.x] = B[tiledRowB * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        // Wait until every thread has loaded its value
        __syncthreads();

        // Multiply the two tiles
        for (int k = 0; k < BLOCK_SIZE; k++)
        {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        // Wait before overwriting shared memory
        __syncthreads();
    }

    if (row < N && col < N)
    {
        C[row * N + col] = sum;
    }
}



__global__ void matrixMulKernel_shared_unrolled(const float* A, const float* B, float* C, int N)
{
    __shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Number of tiles along one matrix dimension
    int numTiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int tile = 0; tile < numTiles; tile++)
    {
        // Global indices to load
        int tiledColA = tile * BLOCK_SIZE + threadIdx.x;
        int tiledRowB = tile * BLOCK_SIZE + threadIdx.y;

        // Load tile from A
        if (row < N && tiledColA < N)
            tileA[threadIdx.y][threadIdx.x] = A[row * N + tiledColA];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile from B
        if (tiledRowB < N && col < N)
            tileB[threadIdx.y][threadIdx.x] = B[tiledRowB * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        // Wait until every thread has loaded its value
        __syncthreads();

        // Multiply the two tiles
        #pragma unroll

        for (int k = 0; k < BLOCK_SIZE; k++)
        {
            sum += tileA[threadIdx.y][k] *
                   tileB[k][threadIdx.x];
        }

        // Wait before overwriting shared memory
        __syncthreads();
    }

    if (row < N && col < N)
    {
        C[row * N + col] = sum;
    }
}