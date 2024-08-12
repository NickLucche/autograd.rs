#pragma once
#include "utils.cuh"

extern "C" {  // need to avoid demangling here
    void testKernelWrapper();

    // template <typename T> cant use templates in C code, no demangling
    std::pair<float *, Dim> conv2dConcurrent(float *dataIn, float *kernel, Dim inDim, Dim kernelDim, Dim stride, int pad, bool alreadyPinned = false);
}