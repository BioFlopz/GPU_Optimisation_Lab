cmake_minimum_required(VERSION 3.18)
project(MATRIX_MULTIPLICATION LANGUAGES CXX CUDA)

# Enforce strict C++17 definitions across host and device codes
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo --ptxas-options=-v")

# Direct CMake to find headers in the include directory
include_directories(include)

# Define executable and append matching source nodes
add_executable(app
    src/main.cu
    src/kernels.cu
)

# Hardcode explicitly to your Turing GTX 1650 silicon capability
set_target_properties(app PROPERTIES
    CUDA_ARCHITECTURES "75"
)
