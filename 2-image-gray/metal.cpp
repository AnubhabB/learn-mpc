#include <iostream>
#include <stdio.h>
#include <fstream>
#include <string>

#include "../imread.h"
using namespace std;

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>

// Load the `.metal` kernel definition as a string
static const char* kernel = {
    #include "imgray.metal"
};

// toGray2D receives
// I, O unsigned char vectors - `_h` is used to denote that these reside in the `host` memory
// I is the input image RGB data in row-major order & O will hold the grayscale converted image
void toGray2D(const unsigned char *I_h, unsigned char *O_h, size_t w, size_t h) {
    // Get the device
    MTL::Device* device = MTLCreateSystemDefaultDevice();
    NS::Error* pError = nullptr;

    // Create a library and lookup the kernel function
    MTL::Library* library = device->newLibrary( NS::String::string(kernel, NS::UTF8StringEncoding), nullptr, &pError );
    MTL::Function* toGray = library->newFunction( NS::String::string("to_gray_kernel", NS::UTF8StringEncoding) );

    // Check if the function was loaded
    if(!toGray) {
        __builtin_printf( "%s", pError->localizedDescription()->utf8String() );
        assert(false);
    }

    // Create a pipeline state
    MTL::ComputePipelineState *_pState = device->newComputePipelineState(toGray, &pError);
    // Create a command queue
    MTL::CommandQueue *cmdQueue = device->newCommandQueue();

    // Calculate the size in bits of the vector -
    // in this case since this `O` is `unsigned char`
    // `I` on the other hand is also `unsigned char` but has 3 channels per pixel - RGB
    const size_t sizeout = w * h * sizeof(unsigned char);
    const size_t sizein = sizeout * 3;

    // Create some new buffers, A_d, B_d are inputs and C_d are the outputs
    MTL::Buffer* I_d = device->newBuffer(I_h, sizein, MTL::ResourceStorageModeManaged);
    MTL::Buffer* O_d = device->newBuffer(sizeout, MTL::ResourceStorageModeManaged);
    MTL::Buffer* W_d = device->newBuffer(&w, sizeof(size_t), MTL::ResourceStorageModeManaged);
    MTL::Buffer* H_d = device->newBuffer(&h, sizeof(size_t), MTL::ResourceStorageModeManaged);

    // Create a command buffer and a compute encoder from the buffer
    MTL::CommandBuffer* cmdBuffer = cmdQueue->commandBuffer();
    MTL::ComputeCommandEncoder* cmdEncoder = cmdBuffer->computeCommandEncoder();
    
    // Set the encoder state and the command buffers
    cmdEncoder->setComputePipelineState(_pState);
    cmdEncoder->setBuffer(I_d, 0, 0);
    cmdEncoder->setBuffer(O_d, 0, 1);
    cmdEncoder->setBuffer(W_d, 0, 2);
    cmdEncoder->setBuffer(H_d, 0, 3);
    
    // Calculate the launch template
    size_t maxthreads = sqrt(_pState->maxTotalThreadsPerThreadgroup());
    MTL::Size threadsPerThreadgroup = MTL::Size::Make(maxthreads, maxthreads, 1);
    MTL::Size threadsPerGrid = MTL::Size::Make(w, h, 1);

    // Execute the kernel
    cmdEncoder->dispatchThreads(threadsPerGrid, threadsPerThreadgroup);
    cmdEncoder->endEncoding();
    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();

    std::memcpy(O_h, O_d->contents(), sizeout);

    toGray->release();
    library->release();
    device->release();
}

int main() {
    // Allocate buffers to input and output image. Input image is 3 channels while output is 1
    unsigned char * img, * out;
    size_t w, h, size;

    // Read image with a helper function
    tie(img, w, h) = imRead();
    
    size = w * h;
    // Allocate data to the output based on the size of the image. Single channel so (w * h * 1)
    out = (unsigned char*)malloc(size * sizeof(unsigned char*));
    // call the kernel caller
    toGray2D(img, out, w, h);
    // helper function to save the result
    gray(out, w, h);
    // Free the allocations
    free(img);
    free(out);
    return 0;
}