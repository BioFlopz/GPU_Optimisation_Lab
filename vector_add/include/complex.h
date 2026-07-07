#ifndef COMPLEX_TYPE_H
#define COMPLEX_TYPE_H

// If compiled by NVCC, use CUDA execution qualifiers. Otherwise, leave them empty for standard C++
#ifdef __CUDACC__
    #define CUDA_HOST_DEVICE __host__ __device__
#else
    #define CUDA_HOST_DEVICE
#endif

struct Complex
{
    float real;
    float imag;

    CUDA_HOST_DEVICE
    Complex(float r = 0.0f, float i = 0.0f)
        : real(r), imag(i) {}

    CUDA_HOST_DEVICE
    Complex operator+(const Complex& b) const
    {
        return Complex(real + b.real,
                       imag + b.imag);
    }

    CUDA_HOST_DEVICE
    Complex operator-(const Complex& b) const
    {
        return Complex(real - b.real,
                       imag - b.imag);
    }

    CUDA_HOST_DEVICE
    Complex operator*(const Complex& b) const
    {
        return Complex(
            real * b.real - imag * b.imag,
            real * b.imag + imag * b.real
        );
    }

    CUDA_HOST_DEVICE
    Complex& operator*=(const Complex& b)
    {
        *this = *this * b;
        return *this;
    }

    CUDA_HOST_DEVICE
    float abs() const
    {
        return sqrtf(real * real + imag * imag);
    }
};


#endif // COMPLEX_TYPE_H