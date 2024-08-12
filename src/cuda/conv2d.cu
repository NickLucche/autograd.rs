#include <iostream>
#include "utils.cuh"

/**
 * Channel last im, col and kernel.
 *
 * assuming kernel [ [1, 2], [3, 4] ] (Cout=1) for RGB input (Cin=3):
 * 1x2x2x3, when reshaped for matmul with ch last col, it will look like:
 *    [1,
 *     1,
 *     1,
 *     2,
 *     2,
 *     2,
 *     3,
 *     3,
 *     3,
 *     4,
 *     4,
 *     4]
 * assuming kernel [ [[1, 2], [3, 4]], [[5, 6], [7, 8]] ] Cout=2, Cin=1:
 * 2x2x2x1, when reshaped for matmul with ch last col, it will look like:
 *    [1, | [5,
 *     2, | 6,
 *     3, | 7,
 *     4] | 8]
 */

template <typename T>
void im2colStreamwise(T *im, T *colOut, Dim imDim, Dim pDim, Dim kernelDim, Dim stride, int pad, int patchOffset, int patchesToCopy, cudaStream_t &stream) {
    // TODO pad
    // im: BxHxWxC
    const auto kkc = kernelDim.w * kernelDim.h * imDim.c;
    const size_t bIdx = patchOffset / (pDim.h * pDim.w);
    const size_t bImOffset = bIdx * imDim.h * imDim.w * imDim.c;
    const size_t initColRow = patchOffset;
    size_t colRow = patchOffset;
    // patch offset wrt current batch (ie batch1, start from patch1)
    patchOffset = patchOffset % (pDim.h * pDim.w);

    // NOTE ch last so you can copy whole 'col' row without huge strides
    // streams will split work *in number of patches* (==rows in "col"); get back y/x abs idxs from patch number
    for (size_t y = patchOffset / pDim.w * stride.h; y <= imDim.h - kernelDim.h; y += stride.h) {
        // skip borders when not enough elems are there (-kw)
        for (size_t x = (patchOffset % pDim.w) * stride.w; x <= imDim.w - kernelDim.w; x += stride.w) {
            // coalesce memory accesses a bit by assigning a whole row of `col`
            // TODO optimize cudaMemcpy3D()
            size_t colIdx = colRow * kkc;
            // NOTE problem, we have to issue multiple memcopies as tiles are not contiguous
            // copy row by row current kernel window
            for (size_t yi = y; yi < y + kernelDim.h; yi++) {
                size_t imIdx = bImOffset + yi * imDim.w + x;
                // copy a kernel row, channel values will be contiguous (last dim)
                // ie R0|G0|B0-R1|G1|B1..
                cudaMemcpyAsync(&colOut[colIdx], &im[imIdx], kernelDim.w * imDim.c * sizeof(T), cudaMemcpyHostToDevice, stream);
                // cudaMemcpy(&colOut[colIdx], &im[imIdx], kernelDim.w * imDim.c * sizeof(T), cudaMemcpyHostToDevice);
                colIdx += kernelDim.w * imDim.c;
            }
            // copied whole "col" row
            colRow++;
            if (colRow - initColRow >= patchesToCopy * imDim.b) return;
        }
    }
}

template <typename T>
// outDim here must be passed by value to be copied automatically onto device!
__global__ void matmul(T *col, T *kernel, T *out, const int n, const Dim outDim, int streamOffset, int streamSize) {
    // NOTE pretend batches are H stacked and you get a bigger image: in memory, as long as you unroll with C order, they look like that
    int h = outDim.h * outDim.b, C_out = outDim.w;
    // n is the size of the shared dim
    // each thread computes one element of the output mat, by accumulating two lines
    // element index in out mat
    int index = blockIdx.x * blockDim.x + threadIdx.x + streamOffset;

    // TODO refactor
    // issue is that each stream will spawn more threads than elems (with small mats) and you need to do check that you only process your chunk
    if (index < h * C_out && index < streamOffset + streamSize) {
        // 1d, this is column index of the thread in grid, which reduces corresponding col in kernel mat
        int kernelCol = index % C_out;
        int colRow = index / C_out;  // row of "col" matrix

        T acc = 0;
        // TODO here we assume kernel is stored in C-order, will be faster in F
        for (size_t k = 0; k < n; k++)
            acc += col[colRow * n + k] * kernel[k * C_out + kernelCol];

        out[index] = acc;
    }
}

template <typename T>
/**
 * @brief Concurrent 2D convolution with im2col and matmul in separate streams.
 * based on https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/overlap-data-transfers/async.cu#L65
 *
 * @param dataIn The input data array. (im) BxHxWxCin.
 * @param kernel The kernel array. KernelDim as CoutxKhxKwxCin. Actual kernel memory layout: KhxKwxCinxCout
 * @param dataOut 
 * @param inDim The dimensions of the input data array.
 * @param kernelDim The dimensions of the kernel array.
 * @param stride The stride value for the convolution operation. Only H,W dims are used.
 * @param pad The padding value for the convolution operation.
 * @param alreadyPinned Flag indicating whether the input data and kernel arrays are already pinned in memory.
 * @return A pair containing the output data array and its dimensions, BxPhxPwxCout.
 */
Dim conv2dConcurrent(T *dataIn, T *kernel, T *dataOut, Dim inDim, Dim kernelDim, Dim stride, int pad, bool alreadyPinned = false) {
    // TODO should I swap kernel first and last dims?
    if (isHostMemory(kernel)) {
        std::cerr << "Kernel must be pre-allocated on device!\n";
        return {nullptr, {0, 0, 0, 0}};
    }
    if (!alreadyPinned) {
        printf("MEM MUST BE PINNED\n");
        return {nullptr, {0, 0, 0, 0}};
    }
    if (inDim.c != kernelDim.c) {
        std::cerr << "Input and kernel channels must match!\n";
        return {nullptr, {0, 0, 0, 0}};
    }
    const int nThreads = 128;
    const int nStreams = 4;
    // NOTE you have to split both in-data copy and matmul, to make sure
    // each stream can operate on the data it copies, split streams on "col" rows==out rows==patches
    const Dim patchDim{inDim.b, (inDim.h + 2 * pad - kernelDim.h) / stride.h + 1, (inDim.w + 2 * pad - kernelDim.w) / stride.w + 1, 1};
    const auto kkc = kernelDim.h * kernelDim.w * inDim.c;
    const Dim colDim{inDim.b, patchDim.h * patchDim.w, kkc, 1};
    const Dim outDim{inDim.b, patchDim.h * patchDim.w, kernelDim.b, 1};  // TODO remove this outDim, work with final dim

    // NOTE for better data locality, each stream gets contiguous patches only (ie NOT same patches across batch dim)
    const int rowsPerStream = (inDim.b * patchDim.h * patchDim.w + nStreams - 1) / nStreams;

    // create streams
    cudaStream_t stream[nStreams];
    for (size_t i = 0; i < nStreams; i++)
        checkCudaErrors(cudaStreamCreate(&stream[i]));

    // im2col->col is B x Pw*Ph x KhKwCin, stream-aware
    T *dataCol;
    checkCudaErrors(cudaMalloc(&dataCol, colDim.size() * sizeof(T)));
    // checkCudaErrors(cudaMallocManaged(&dataCol, colDim.size() * sizeof(T)));

    for (size_t i = 0; i < nStreams; i++) {
        auto rOffset = i * rowsPerStream;
        // copy conv tiles to GPU (async)
        im2colStreamwise<T>(dataIn, dataCol, inDim, patchDim, kernelDim, stride, pad, rOffset, rowsPerStream, stream[i]);
    }

    // actual output of conv, Pw*Ph x C_out (num filters)
    T *dataOut;
    checkCudaErrors(cudaMalloc(&dataOut, outDim.size() * sizeof(T)));
    T *dataOuthost = new T[outDim.size()];

    // single thread computes single element of final mat
    const auto C_out = kernelDim.b;
    const int streamElems = rowsPerStream * C_out;
    // further divide work on each stream (ie M patches/rows) evenly among blocks
    // NOTE kernel mat is shared across streams (multiple threads will read same kernel column)
    const int nBlocks = (streamElems + nThreads - 1) / nThreads;
    for (size_t i = 0; i < nStreams; i++) {
        // assign rows to streams like we did in im2col, but now C_out is a free dim
        auto offset = i * streamElems;
        // Pw*Ph x KhKwCin @ KhKwCin x C_out =  Pw*Ph x  C_out
        matmul<T><<<nBlocks, nThreads, 0, stream[i]>>>(dataCol, kernel, dataOut, kkc, outDim, offset, streamElems);
        // stream chunk D2H, can cross multiple batches, but all patches are always contiguous=>we only need a single memcpy
        cudaMemcpyAsync(&dataOuthost[offset], &dataOut[offset], streamElems * sizeof(T), cudaMemcpyDeviceToHost, stream[i]);
    }
    cudaDeviceSynchronize();
    // "reshape"
    // avoid std::pair due to undefined layout rust FFI warning
    // return {dataOuthost, {inDim.b, patchDim.h, patchDim.w, kernelDim.b}};
}

// c++ compiler does not generate the compiled object code for a template function or class until it encounters
// an explicit instantiated template during the compilation phase, so it will error out in linking
template std::pair<float *, Dim> conv2dConcurrent<float>(float *, float *, Dim, Dim, Dim, int, bool);