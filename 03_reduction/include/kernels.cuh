#pragma once

#include <cuda_runtime.h>

__global__ void reduceNaive(const float* input, float* output, int n);