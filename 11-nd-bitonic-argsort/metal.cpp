#include <iostream>
#include <stdio.h>
#include <fstream>
#include <string>

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>

using namespace std;

// Load the `.metal` kernel definition as a string
static const char* kernel = {
    #include "bas.metal"
};

int main() {
    const size_t nrows = 1;
    const size_t ncols = 2048;
    size_t ncols_pad = 1;
    while(ncols_pad < ncols)
        ncols_pad *= 2;

    // Get the device
    MTL::Device* device = MTLCreateSystemDefaultDevice();
    NS::Error* pError = nullptr;

    // Create a library and lookup the kernel function
    MTL::Library* library = device->newLibrary( NS::String::string(kernel, NS::UTF8StringEncoding), nullptr, &pError );
    MTL::Function* bsort = library->newFunction( NS::String::string("bitonicArgSort", NS::UTF8StringEncoding) );

    // Check if the function was loaded
    if(!bsort) {
        __builtin_printf( "%s", pError->localizedDescription()->utf8String() );
        assert(false);
    }

    // Create a pipeline state
    MTL::ComputePipelineState *_pState = device->newComputePipelineState(bsort, &pError);
    // Create a command queue
    MTL::CommandQueue *cmdQueue = device->newCommandQueue();

    float *X_cont = (float *)malloc(nrows * ncols * sizeof(float));
    for(int i=0; i < nrows*ncols; i++) {
        X_cont[i] = static_cast<float>(i);
    }

    MTL::Buffer *X = device->newBuffer(X_cont, nrows * ncols * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer *C_d = device->newBuffer(nrows * ncols * sizeof(int), MTL::ResourceStorageModeManaged);
    MTL::Buffer *N_COLS = device->newBuffer(&ncols, sizeof(size_t), MTL::ResourceStorageModePrivate);
    MTL::Buffer *N_COLS_PAD = device->newBuffer(&ncols_pad, sizeof(size_t), MTL::ResourceStorageModePrivate);
    // MTL::Buffer *X_ = device->newBuffer(&X, sizeof(size_t), MTL::ResourceStorageModePrivate);

    // Create a command buffer and a compute encoder from the buffer
    MTL::CommandBuffer* cmdBuffer = cmdQueue->commandBuffer();
    MTL::ComputeCommandEncoder* cmdEncoder = cmdBuffer->computeCommandEncoder();

    // Set the encoder state and the command buffers
    cmdEncoder->setComputePipelineState(_pState);
    cmdEncoder->setBuffer(X, 0, 0);
    cmdEncoder->setBuffer(C_d, 0, 1);
    cmdEncoder->setBuffer(N_COLS, 0, 2);
    cmdEncoder->setBuffer(N_COLS_PAD, 0, 3);
    // cmdEncoder->setThreadgroupMemoryLength(ncols_pad * sizeof(uint32_t), 0);

    // cmdEncoder->setBuffer(B, 0, 1);
    // cmdEncoder->setBuffer(C_d, 0, 2);
    // cmdEncoder->setBuffer(M_, 0, 3);
    // cmdEncoder->setBuffer(N_, 0, 4);
    // cmdEncoder->setBuffer(P_, 0, 5);

    // Calculate the launch template
    // size_t maxThreadsPerThreadgroup = _pState->maxTotalThreadsPerThreadgroup();
    // // Calculate optimal thread group size
    // size_t threadGroupWidth = std::min(static_cast<size_t>(32), P);
    // size_t threadGroupHeight = std::min(static_cast<size_t>(ceil(maxThreadsPerThreadgroup / threadGroupWidth)), M);
    // MTL::Size threadsPerThreadgroup = MTL::Size::Make(threadGroupWidth, threadGroupHeight, 1);

    // Calculate grid size
    // size_t gridWidth = ceil((P + threadGroupWidth - 1) / threadGroupWidth);
    // size_t gridHeight = ceil((M + threadGroupHeight - 1) / threadGroupHeight);
    // MTL::Size gridSize = MTL::Size::Make(gridWidth, gridHeight, 1);

    // Execute the kernel
    cmdEncoder->dispatchThreadgroups(
        MTL::Size::Make(static_cast<uint64_t>(ceil(static_cast<float>(ncols_pad) / 1024.)), nrows, 1),
        MTL::Size::Make(1024, 1, 1)
    );
    cmdEncoder->endEncoding();
    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();

    printf("Max shared: %lu\n", _pState->staticThreadgroupMemoryLength());
    uint * t = static_cast<uint*>(C_d->contents());
    for(int i=0; i<nrows * ncols;i++) {
        printf("%u ", t[i]);
    }
    // uint32_t* result = (uint32_t*)malloc(nrows * ncols * sizeof(uint32_t));
    // memcpy(result, static_cast<uint32_t*>(C_d->contents()), nrows * ncols * sizeof(uint32_t));

    bsort->release();

    // MTL::Buffer *A = device->newBuffer(M * N * sizeof(float), MTL::ResourceStorageModeShared);
    // MTL::Buffer *B = device->newBuffer(N * P * sizeof(float), MTL::ResourceStorageModeShared);
    // MTL::Buffer *C_d = device->newBuffer(M * P * sizeof(float), MTL::ResourceStorageModeShared);
    // MTL::Buffer *M_ = device->newBuffer(&M, sizeof(size_t), MTL::ResourceStorageModePrivate);
    // MTL::Buffer *N_ = device->newBuffer(&N, sizeof(size_t), MTL::ResourceStorageModePrivate);
    // MTL::Buffer *P_ = device->newBuffer(&P, sizeof(size_t), MTL::ResourceStorageModePrivate);

    // MTL::Function* coalesced = library->newFunction( NS::String::string("coalesced", NS::UTF8StringEncoding) );

    // // Check if the function was loaded
    // if(!naive) {
    //     __builtin_printf( "%s", pError->localizedDescription()->utf8String() );
    //     assert(false);
    // }

    library->release();
    device->release();

    return 0;
}