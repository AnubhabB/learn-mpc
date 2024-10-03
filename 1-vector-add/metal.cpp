#include <iostream>
#include <stdio.h>
#include <fstream>
#include <string>
using namespace std;

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>

// Load the `.metal` kernel definition as a string
static const char* kernel = {
    #include "vectoradd.metal"
};

// vecAdd receives
// A, B, C are pointers to float vectors - `_h` is used to denote that these reside in the `host` memory
// A & B are the input data and C will hold the addition output
void vecAdd(float *A_h, float *B_h, float *C_h, int n) {
    // Get the device
    MTL::Device* device = MTLCreateSystemDefaultDevice();
    NS::Error* pError = nullptr;

    // Create a library and lookup the kernel function
    MTL::Library* library = device->newLibrary( NS::String::string(kernel, NS::UTF8StringEncoding), nullptr, &pError );
    MTL::Function* vecAdd = library->newFunction( NS::String::string("vec_add", NS::UTF8StringEncoding) );

    // Check if the function was loaded
    if(!vecAdd) {
        __builtin_printf( "%s", pError->localizedDescription()->utf8String() );
        assert(false);
    }

    // Create a pipeline state
    MTL::ComputePipelineState *_pState = device->newComputePipelineState(vecAdd, &pError);
    // Create a command queue
    MTL::CommandQueue *cmdQueue = device->newCommandQueue();

    // Calculate the size in bits of the vector -
    // in this case since this is a `single precision float` we are effectively saying `n * 4 bytes` or `n * 32 bits`
    const size_t size = n * sizeof(float);

    // Create some new buffers, A_d, B_d are inputs and C_d are the outputs
    MTL::Buffer* A_d = device->newBuffer(A_h, size, MTL::ResourceStorageModeShared);
    MTL::Buffer* B_d = device->newBuffer(B_h, size, MTL::ResourceStorageModeShared);
    MTL::Buffer* C_d = device->newBuffer(size, MTL::ResourceStorageModeShared);
    MTL::Buffer* N = device->newBuffer(&n, sizeof(n), MTL::ResourceStorageModeShared);
    

    // Create a command buffer and a compute encoder from the buffer
    MTL::CommandBuffer* cmdBuffer = cmdQueue->commandBuffer();
    MTL::ComputeCommandEncoder* cmdEncoder = cmdBuffer->computeCommandEncoder();
    
    // Set the encoder state and the command buffers
    cmdEncoder->setComputePipelineState(_pState);
    cmdEncoder->setBuffer(A_d, 0, 0);
    cmdEncoder->setBuffer(B_d, 0, 1);
    cmdEncoder->setBuffer(C_d, 0, 2);
    cmdEncoder->setBuffer(N, 0, 3);
    
    // Calculate the launch template
    MTL::Size gridSize = MTL::Size::Make(n, 1, 1);
    size_t tgs = _pState->maxTotalThreadsPerThreadgroup();
    if ( tgs > n ) {
        tgs = n;
    }
    MTL::Size threadGroupSize = MTL::Size::Make(tgs, 1, 1);

    // Execute the kernel
    cmdEncoder->dispatchThreads(gridSize, threadGroupSize);
    cmdEncoder->endEncoding();
    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();

    std::memcpy(C_h, C_d->contents(), size);
    

    vecAdd->release();
    library->release();
    device->release();
}

int main() {

    float *A, *B, *C;
    const int size = 2048;
    
    A = (float*)malloc(size * sizeof(float));
    B = (float*)malloc(size * sizeof(float));
    C = (float*)malloc(size * sizeof(float));

    for (int i = 0; i < size; i++) {
        A[i] = 2.0;
        B[i] = 3.0;
    }

    vecAdd(A, B, C, size);

    // Check if all is well
    for (int i=0; i < size; i ++) {
        if ( C[i] != 5.0 ) {
            printf("Error! %f %f %f", A[i], B[i], C[i]);
            return 1;
        }
    }

    printf("Checks passed!");

    free(A);
    free(B);
    free(C);

    return 0;
}