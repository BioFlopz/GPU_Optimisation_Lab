#pragma once

#include <cuda_runtime.h>

__global__ void reduceNaive(const float* input, float* output, int n);
__global__ void reduce_interleaved(const float* input, float* output, int n);
__global__ void reduce_v3(const float* input, float* output, int n);
__global__ void reduce_v4(const float* input, float* output, int n);
__global__ void reduce_v5(const float* input, float* output, int n);
