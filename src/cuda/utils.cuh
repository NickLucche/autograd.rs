#pragma once
#include <iostream>
#define checkCudaErrors(call)                                       \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err));                        \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)
    
struct Dim {
    int b = 1;
    int h;
    int w;
    int c;
    size_t size() const { return b * h * w * c; }
    // __host__ __device__
    Dim(int b = 1, int h = 1, int w = 1, int c = 1) : b(b), h(h), w(w), c(c) {}

    friend std::ostream &operator<<(std::ostream &os, const Dim &dim);
};

inline std::ostream &operator<<(std::ostream &os, const Dim &dim) {
    os << "B: " << dim.b << ", H: " << dim.h << ", W: " << dim.w << ", C: " << dim.c << std::endl;
    return os;
}


inline bool isHostMemory(void *ptr) {
    cudaPointerAttributes attributes;
    cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);

    if (err != cudaSuccess) {
        std::cerr << "Error getting pointer attributes: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    // Note: cudaPointerAttributes::memoryType is deprecated in CUDA 11.0 and later,
    // use cudaPointerAttributes::type instead.
#if CUDART_VERSION >= 10000
    return attributes.type == cudaMemoryTypeHost;
#else
    return attributes.memoryType == cudaMemoryTypeHost;
#endif
}

int getNumberOfSMs() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    return props.multiProcessorCount;
}

int getNumberOfBlocks(int totalElements, int threadsPerBlock) {
    // put function here if you want to play around with smarter policies for finding right number of blocks
    return (totalElements + threadsPerBlock - 1) / threadsPerBlock;
}