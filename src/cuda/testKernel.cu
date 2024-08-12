#include <iostream>
#include "conv2d.h"

#define checkCudaErrors(call)                                       \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err));                        \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

__global__ void testKernel(float *data, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) data[index] = 1;
}

void testKernelWrapper() {
    printf("testKernelWrapper\n");
    float *data;
    int n = 10;
    checkCudaErrors(cudaMallocManaged(&data, n * sizeof(float)));
    for (size_t i = 0; i < n; i++) data[i] = i * 2;
    for (size_t i = 0; i < n; i++) printf("%f ", data[i]);
    printf("\n");

    testKernel<<<(n + 127) / 128, 128>>>(data, n);
    // pick up invalid args like blockSize too big! (ie when you pass 4kthreads per block)
    checkCudaErrors(cudaPeekAtLastError());
    cudaDeviceSynchronize();
    for (size_t i = 0; i < n; i++) printf("%f ", data[i]);
    printf("\n");
}
