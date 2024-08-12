#include "utils.cuh"
#define NTHREADS 256

// TODO just basic implementations for now
template <typename T>
__global__ void add(T *a, T *b, T *out, Dim dim) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO grid stride loop
    if (idx < dim.size()) {        
        out[idx] = a[idx] + b[idx];
    }
}


template <typename T>
T* addHandler(T *a, T* b, Dim dim, bool inplace) {
    // TODO broadcast rule on Channel dim
    T *c = a;
    if (!inplace)
        checkCudaErrors(cudaMalloc(&c, dim.size() * sizeof(T)));

    add<<<getNumberOfBlocks(dim.size(), NTHREADS), NTHREADS>>>(a, b, c, dim);
    checkCudaErrors(cudaDeviceSynchronize());

    return c;
}

// c++ compiler does not generate the compiled object code for a template function or class until it encounters
// an explicit instantiated template during the compilation phase, so it will error out in linking
template float* addHandler<float>(float *, float *, Dim, bool);