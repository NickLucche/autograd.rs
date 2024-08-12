#include <cudnn.h>

#include <cassert>
#include <cmath>
#include <vector>

#include "utils.cuh"
#define CHECK_CUDNN(expression)                                    \
    {                                                              \
        cudnnStatus_t status = (expression);                       \
        if (status != CUDNN_STATUS_SUCCESS) {                      \
            std::cerr << "Error on line " << __LINE__ << ": "      \
                      << cudnnGetErrorString(status) << std::endl; \
            std::exit(EXIT_FAILURE);                               \
        }                                                          \
    }

template <typename T>
extern std::pair<T *, Dim> conv2dConcurrent(T *, T *, Dim, Dim, Dim, int, bool);

void testConv2dConcurrent() {
    // allocate pinned host memory for the image to memcpy async
    float *img, *kernel;
    Dim kernelDim{1, 2, 2, 1};                           // KhxKwxC_out
    Dim imDim{1, 3, 3, 1};                               // HxWxC_in
    cudaMallocHost(&img, imDim.size() * sizeof(float));  // pinned
    // cudaMalloc(&kernel, kernelDim.size() * sizeof(float));
    cudaMallocManaged(&kernel, kernelDim.size() * sizeof(float));
    printf("Input:\n");
    for (size_t i = 0; i < imDim.h; i++) {
        for (size_t j = 0; j < imDim.w; j++) {
            auto idx = i * imDim.w + j;
            if (i == 0 || i == 1 && j == 1 || i == 2 && j == 2)
                img[idx] = 1;
            printf("%f ", img[idx]);
        }
        printf("\n");
    }
    printf("\n");

    // NOTE allocate kernel with matmul friendly layout
    printf("Kernel 2D:\n");
    for (size_t i = 0; i < kernelDim.h; i++) {
        for (size_t j = 0; j < kernelDim.w; j++) {
            auto idx = i * kernelDim.w + j;
            if (i == 0 && (j == 0 || j == 2) || i == 1 && j == 1)
                kernel[idx] = 1;
            else if (i == 1 && j == 0 || i == 2 && j == 1)
                kernel[idx] = -1;
            printf("%f ", kernel[idx]);
        }
        printf("\n");
    }
    printf("\n");

    Dim stride{1, 1, 1, 1};
    auto [out, outDim] = conv2dConcurrent(img, kernel, imDim, kernelDim, stride, 0, true);

    printf("Output:\n");
    for (size_t i = 0; i < outDim.h; i++) {
        for (size_t j = 0; j < outDim.w; j++) {
            auto idx = i * outDim.w + j;
            printf("%f ", out[idx]);
        }
        printf("\n");
    }
    printf("\n");

    cudaFree(img);
    cudaFree(kernel);
    free(out);
}

// TODO gtest
void testConv2dConcurrentBatched() {
    // same test, but with batched input, 2 images and 3 kernels

    // TODO factor out in
    // allocate pinned host memory for the image to memcpy async
    float *img, *kernel;
    Dim kernelDim{3, 2, 2, 1};                           // C_outxKhxKwxC_in
    Dim imDim{2, 3, 3, 1};                               // HxWxC_in
    cudaMallocHost(&img, imDim.size() * sizeof(float));  // pinned
    // cudaMalloc(&kernel, kernelDim.size() * sizeof(float));
    cudaMallocManaged(&kernel, kernelDim.size() * sizeof(float));
    printf("Input:\n");
    for (size_t b = 0; b < imDim.b; b++) {
        for (size_t i = 0; i < imDim.h; i++) {
            for (size_t j = 0; j < imDim.w; j++) {
                auto idx = b * (imDim.h * imDim.w) + i * imDim.w + j;
                if (i == 0 || i == 1 && j == 1 || i == 2 && j == 2)
                    img[idx] = 1 * (b == 0) ? 1 : 2;
                printf("%f ", img[idx]);
            }
            printf("\n");
        }
        printf("+++++++++++++++\n");
    }
    printf("\n");

    // easy to read kernel
    std::vector<std::vector<float>> kernelV = {
        {1, 0,
         -1, 1},
        {2, 0,
         -2, 2},
        {2, 0,
         -2, 2}};
    // allocate kernel with matmul friendly layout
    size_t kvalIdx = 0;
    for (size_t krow = 0; krow < kernelDim.h * kernelDim.w * imDim.c; krow += imDim.c) {
        for (size_t filterNo = 0; filterNo < kernelDim.b; filterNo++) {
            // replicate kernel value for each channel Cin (channel last im)
            for (size_t c = 0; c < imDim.c; c++) {
                auto idx = (krow + c) * kernelDim.b + filterNo;
                kernel[idx] = kernelV[filterNo][kvalIdx];
            }
        }
        kvalIdx++;
    }

    printf("Kernel KKCxC_out:\n");
    for (size_t krow = 0; krow < kernelDim.h * kernelDim.w * imDim.c; krow++) {
        for (size_t filterNo = 0; filterNo < kernelDim.b; filterNo++) {
            auto idx = krow * kernelDim.b + filterNo;
            printf("%f ", kernel[idx]);
        }
        printf("\n");
    }
    printf("\n");

    Dim stride{1, 1, 1, 1};
    auto [out, outDim] = conv2dConcurrent(img, kernel, imDim, kernelDim, stride, 0, true);

    std::cout << "OutDim " << outDim << std::endl;
    for (size_t b = 0; b < outDim.b; b++) {
        for (size_t i = 0; i < outDim.h; i++) {
            for (size_t j = 0; j < outDim.w; j++) {
                for (size_t c = 0; c < outDim.c; c++) {
                    auto idx = b * (outDim.h * outDim.w * outDim.c) + i * outDim.w * outDim.c + j * outDim.c + c;
                    printf("%f ", out[idx]);
                }
                printf(" | ");
            }
            printf("\n");
        }
        printf("+++++++++++++++\n");
    }
    printf("\n");

    cudaFree(img);
    cudaFree(kernel);
    free(out);
}

void testConv2dConcurrentMultiInChannel() {
}

void testConv2dConcurrentcuDNN() {
    float *inputData;
    float *filterData;
    float *outputData;
    // hardcoded sizes
    int batchSize = 2, inputHeight = 3, inputWidth = 3, inputChannels = 1;

    int outputHeight = 2, outputWidth = 2, outputChannels = 3;
    int filterHeight = 2, filterWidth = 2;

    // Create cudnn tensors (descriptors)
    cudnnTensorDescriptor_t inputDescriptor;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&inputDescriptor));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(inputDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, inputChannels, inputHeight, inputWidth));

    cudnnFilterDescriptor_t filterDescriptor;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filterDescriptor));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filterDescriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, outputChannels, inputChannels, filterHeight, filterWidth));

    cudnnConvolutionDescriptor_t convolutionDescriptor;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convolutionDescriptor));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convolutionDescriptor, /*pad_height=*/0, /*pad_width=*/0, /*vertical_stride=*/1, /*horizontal_stride=*/1, /*dilation_height=*/1, /*dilation_width=*/1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    cudnnTensorDescriptor_t outputDescriptor;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&outputDescriptor));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(outputDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, outputChannels, outputHeight, outputWidth));

    // Allocate memory on device
    cudaMalloc(&inputData, sizeof(float) * batchSize * inputChannels * inputHeight * inputWidth);
    cudaMalloc(&filterData, sizeof(float) * outputChannels * inputChannels * filterHeight * filterWidth);
    cudaMalloc(&outputData, sizeof(float) * batchSize * outputChannels * outputHeight * outputWidth);
    // cudaMemset(outputData, 0, sizeof(float) * batchSize * outputChannels * outputHeight * outputWidth);

    // Get the output dimensions
    int outputDims = 4;
    int outputDimA[4];
    cudnnGetConvolutionNdForwardOutputDim(convolutionDescriptor, inputDescriptor, filterDescriptor, outputDims, outputDimA);

    // Print the output dimensions
    printf("Output Dimensions: ");
    for (int i = 0; i < outputDims; i++) {
        printf("%d ", outputDimA[i]);
    }
    printf("\n");

    // copy actual data values
    float inputValues[] = {1., 1., 1., 0., 1., 0., 0., 0., 1.,
                           2., 2., 2., 0., 2., 0., 0., 0., 2.};  // batch_0 | batch_1
    cudaMemcpy(inputData, inputValues, sizeof(float) * batchSize * inputChannels * inputHeight * inputWidth, cudaMemcpyHostToDevice);

    // float filterValues[] = {1., 0., -1., 1.};
    float filterValues[] = {1., 0., -1., 1., 2., 0., -2., 2., 2., 0., -2., 2.};
    cudaMemcpy(filterData, filterValues, sizeof(float) * outputChannels * inputChannels * filterHeight * filterWidth, cudaMemcpyHostToDevice);

    // Initialize cuDNN
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    // Perform convolution using cuDNN
    float alpha = 1.0f, beta = 0.0f;
    // size_t workspaceSize;
    // allocate workspace for cuDNN https://docs.nvidia.com/deeplearning/cudnn/latest/api/cudnn-cnn-library.html#cudnnconvolutionforward
    // CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, inputDescriptor, filterDescriptor, convolutionDescriptor, outputDescriptor, CUDNN_CONVOLUTION_FWD_ALGO_GEMM, &workspaceSize));
    // void *workspace;
    // cudaMalloc(&workspace, workspaceSize);
    CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha, inputDescriptor, inputData, filterDescriptor, filterData, convolutionDescriptor, CUDNN_CONVOLUTION_FWD_ALGO_GEMM, nullptr, 0, &beta, outputDescriptor, outputData));
    // cudaFree(workspace);

    float *outputValues = new float[batchSize * outputChannels * outputHeight * outputWidth];
    cudaMemcpy(outputValues, outputData, sizeof(float) * batchSize * outputChannels * outputHeight * outputWidth, cudaMemcpyDeviceToHost);

    std::cout << "Output Data:" << std::endl;
    for (int b = 0; b < batchSize; b++) {
        for (int c = 0; c < outputChannels; c++) {
            for (int i = 0; i < outputHeight; i++) {
                for (int j = 0; j < outputWidth; j++) {
                    // channel last
                    int idx = b * (outputHeight * outputWidth * outputChannels) + c * (outputWidth * outputHeight) + i * outputWidth + j;
                    std::cout << outputValues[idx] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "-------------\n";  // ch/filter activation delimiter
        }
        std::cout << "+++++++++++++++" << std::endl;  // batch delimiter
    }
    std::cout << std::endl;

    delete[] outputValues;

    // Perform convolution using conv2dConcurrent
    // float* concurrentOutputData;
    // // Allocate memory for concurrentOutputData...
    // conv2dConcurrent(inputData, filterData, concurrentOutputData, batchSize, inputHeight, inputWidth, inputChannels, outputHeight, outputWidth, outputChannels, filterHeight, filterWidth);

    // // Compare the results
    // for (int i = 0; i < batchSize * outputHeight * outputWidth * outputChannels; i++) {
    //     assert(std::abs(outputData[i] - concurrentOutputData[i]) < 1e-6);
    // }

    cudnnDestroyTensorDescriptor(inputDescriptor);
    cudnnDestroyFilterDescriptor(filterDescriptor);
    cudnnDestroyConvolutionDescriptor(convolutionDescriptor);
    cudnnDestroyTensorDescriptor(outputDescriptor);
    cudnnDestroy(cudnn);
    cudaFree(inputData);
    cudaFree(outputData);
    cudaFree(filterData);
}

int main() {
    testConv2dConcurrent();
    testConv2dConcurrentBatched();
    testConv2dConcurrentcuDNN();
    return 0;
}