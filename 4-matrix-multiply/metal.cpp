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
    #include "mm.metal"
};

int main() {
    const size_t M = 2048;
    const size_t N = 4096;
    const size_t P = 1024;

    // Get the device
    MTL::Device* device = MTLCreateSystemDefaultDevice();
    NS::Error* pError = nullptr;

    // Create a library and lookup the kernel function
    MTL::Library* library = device->newLibrary( NS::String::string(kernel, NS::UTF8StringEncoding), nullptr, &pError );
    MTL::Function* naive = library->newFunction( NS::String::string("naive", NS::UTF8StringEncoding) );

    // Check if the function was loaded
    if(!naive) {
        __builtin_printf( "%s", pError->localizedDescription()->utf8String() );
        assert(false);
    }

    // Create a pipeline state
    MTL::ComputePipelineState *_pState = device->newComputePipelineState(naive, &pError);
    // Create a command queue
    MTL::CommandQueue *cmdQueue = device->newCommandQueue();

    MTL::Buffer *A = device->newBuffer(M * N * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer *B = device->newBuffer(N * P * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer *C_d = device->newBuffer(M * P * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer *M_ = device->newBuffer(&M, sizeof(size_t), MTL::ResourceStorageModePrivate);
    MTL::Buffer *N_ = device->newBuffer(&N, sizeof(size_t), MTL::ResourceStorageModePrivate);
    MTL::Buffer *P_ = device->newBuffer(&P, sizeof(size_t), MTL::ResourceStorageModePrivate);

    float *A_cont = static_cast<float*>(A->contents());
    float *B_cont = static_cast<float*>(B->contents());
    for(int i=0; i < M*N; i++) {
        A_cont[i] = static_cast<float>(i);
        if(i < N*P) {
            B_cont[i] = static_cast<float>(i);
        }
    }

    // Create a command buffer and a compute encoder from the buffer
    MTL::CommandBuffer* cmdBuffer = cmdQueue->commandBuffer();
    MTL::ComputeCommandEncoder* cmdEncoder = cmdBuffer->computeCommandEncoder();

    // Set the encoder state and the command buffers
    cmdEncoder->setComputePipelineState(_pState);
    cmdEncoder->setBuffer(A, 0, 0);
    cmdEncoder->setBuffer(B, 0, 1);
    cmdEncoder->setBuffer(C_d, 0, 2);
    cmdEncoder->setBuffer(M_, 0, 3);
    cmdEncoder->setBuffer(N_, 0, 4);
    cmdEncoder->setBuffer(P_, 0, 5);

    // Calculate the launch template
    size_t maxThreadsPerThreadgroup = _pState->maxTotalThreadsPerThreadgroup();
    // Calculate optimal thread group size
    size_t threadGroupWidth = std::min(static_cast<size_t>(32), P);
    size_t threadGroupHeight = std::min(static_cast<size_t>(ceil(maxThreadsPerThreadgroup / threadGroupWidth)), M);
    MTL::Size threadsPerThreadgroup = MTL::Size::Make(threadGroupWidth, threadGroupHeight, 1);

    // Calculate grid size
    size_t gridWidth = ceil((P + threadGroupWidth - 1) / threadGroupWidth);
    size_t gridHeight = ceil((M + threadGroupHeight - 1) / threadGroupHeight);
    MTL::Size gridSize = MTL::Size::Make(gridWidth, gridHeight, 1);

    // Execute the kernel
    cmdEncoder->dispatchThreadgroups(gridSize, threadsPerThreadgroup);
    cmdEncoder->endEncoding();
    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();

    float* result = (float*)malloc(M * P * sizeof(float));
    memcpy(result, static_cast<float*>(C_d->contents()), M * P * sizeof(float));

    naive->release();

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