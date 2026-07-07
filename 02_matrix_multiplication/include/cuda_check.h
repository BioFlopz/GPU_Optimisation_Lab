#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define CHECK_CUDA(call)                                                   \
do {                                                                        \
    cudaError_t err = call;                                                 \
    if (err != cudaSuccess)                                                 \
    {                                                                       \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << '\n';     \
        std::cerr << "File: " << __FILE__ << '\n';                          \
        std::cerr << "Line: " << __LINE__ << '\n';                          \
        std::exit(EXIT_FAILURE);                                            \
    }                                                                       \
} while (0)