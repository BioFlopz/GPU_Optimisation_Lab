#pragma once

#include <cuda_runtime.h>

__global__ void matrixMulKernel(const float* A, const float* B, float* C, int N);

__global__ void matrixMulKernel_shared(const float* A, const float* B, float* C, int N);
__global__ void matrixMulKernel_shared_unrolled(const float* A, const float* B, float* C, int N);