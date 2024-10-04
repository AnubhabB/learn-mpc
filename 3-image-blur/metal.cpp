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
    #include "imblur.metal"
};

void naiveBlur(
    const unsigned char* I_h,
    unsigned char* O_h,
    const size_t w,
    const size_t h,
    const int radius,
    const unsigned char chan
) {
    // Get the device
    MTL::Device* device = MTLCreateSystemDefaultDevice();
    NS::Error* pError = nullptr;

    // Create a library and lookup the kernel function
    MTL::Library* library = device->newLibrary( NS::String::string(kernel, NS::UTF8StringEncoding), nullptr, &pError );
    MTL::Function* toBlur = library->newFunction( NS::String::string("to_blur_kernel", NS::UTF8StringEncoding) );

    // Check if the function was loaded
    if(!toBlur) {
        __builtin_printf( "%s", pError->localizedDescription()->utf8String() );
        assert(false);
    }

    // Create a pipeline state
    MTL::ComputePipelineState *_pState = device->newComputePipelineState(toBlur, &pError);
    // Create a command queue
    MTL::CommandQueue *cmdQueue = device->newCommandQueue();

    // Calculate the size in bits of the vector -
    // in this case since this `O` is `unsigned char`
    // `I` on the other hand is also `unsigned char` but has 3 channels per pixel - RGB
    const size_t true_w = w * chan;
    const size_t size = true_w * h * sizeof(unsigned char);
    

    // Create some new buffers, A_d, B_d are inputs and C_d are the outputs
    MTL::Buffer* I_d = device->newBuffer(I_h, size, MTL::ResourceStorageModeManaged);
    MTL::Buffer* O_d = device->newBuffer(size, MTL::ResourceStorageModeManaged);
    MTL::Buffer* W_d = device->newBuffer(&true_w, sizeof(size_t), MTL::ResourceStorageModeManaged);
    MTL::Buffer* H_d = device->newBuffer(&h, sizeof(size_t), MTL::ResourceStorageModeManaged);
    MTL::Buffer* R_d = device->newBuffer(&radius, sizeof(int), MTL::ResourceStorageModeManaged);
    MTL::Buffer* C_d = device->newBuffer(&chan, sizeof(unsigned char), MTL::ResourceStorageModeManaged);

    // Create a command buffer and a compute encoder from the buffer
    MTL::CommandBuffer* cmdBuffer = cmdQueue->commandBuffer();
    MTL::ComputeCommandEncoder* cmdEncoder = cmdBuffer->computeCommandEncoder();
    
    // Set the encoder state and the command buffers
    cmdEncoder->setComputePipelineState(_pState);
    cmdEncoder->setBuffer(I_d, 0, 0);
    cmdEncoder->setBuffer(O_d, 0, 1);
    cmdEncoder->setBuffer(W_d, 0, 2);
    cmdEncoder->setBuffer(H_d, 0, 3);
    cmdEncoder->setBuffer(R_d, 0, 4);
    cmdEncoder->setBuffer(C_d, 0, 5);
    
    // Calculate the launch template
    size_t maxThreadsPerThreadgroup = _pState->maxTotalThreadsPerThreadgroup();

    // Calculate optimal thread group size
    size_t threadGroupWidth = std::min(static_cast<size_t>(32), true_w);
    size_t threadGroupHeight = std::min(ceil(maxThreadsPerThreadgroup / threadGroupWidth), h);
    MTL::Size threadsPerThreadgroup = MTL::Size::Make(threadGroupWidth, threadGroupHeight, 1);

    // Calculate grid size
    size_t gridWidth = ceil((true_w + threadGroupWidth - 1) / threadGroupWidth);
    size_t gridHeight = ceil((h + threadGroupHeight - 1) / threadGroupHeight);
    MTL::Size gridSize = MTL::Size::Make(gridWidth, gridHeight, 1);

    // Execute the kernel
    cmdEncoder->dispatchThreadgroups(gridSize, threadsPerThreadgroup);
    cmdEncoder->endEncoding();
    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();

    std::memcpy(O_h, O_d->contents(), size);

    toBlur->release();
    library->release();
    device->release();
}

int main() {
    const int RADIUS = 64;
    // Allocate buffers to input and output image. Input image is 3 channels while output is 1
    unsigned char *img, *out, chan;
    size_t w, h, size;

    // Read image with a helper function
    tie(img, w, h, chan) = imRead("data/gray-lion.jpg");

    size = w * h;
    printf("Gray lion: %lu %lu %d\n", w, h, chan);
    // Allocate data to the output based on the size of the image. Single channel so (w * h * 1)
    out = (unsigned char*)malloc(size * chan * sizeof(unsigned char));
    // call the kernel caller
    naiveBlur(img, out, w, h, RADIUS, chan);
    // helper function to save the result
    jpeg_write(out, w, h, chan, "data/blur-lion-grey-m.jpg");
    // Free the allocations
    free(img);
    free(out);

    // Read image with a helper function
    tie(img, w, h, chan) = imRead("data/lion.jpg");

    size = w * h;
    printf("RGB lion: %lu %lu %d\n", w, h, chan);
    // Allocate data to the output based on the size of the image. Single channel so (w * h * 1)
    out = (unsigned char*)malloc(size * chan * sizeof(unsigned char));
    // call the kernel caller
    naiveBlur(img, out, w, h, RADIUS, chan);
    // helper function to save the result
    jpeg_write(out, w, h, chan, "data/blur-lion-m.jpg");
    // Free the allocations
    free(img);
    free(out);
    return 0;
}