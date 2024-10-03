#include "../imread.h"

// A kernel to convert RGB image to gray scale using the formula:
// Gray[Y, X] = r[y,x] * 0.21 + g[y, x] * 0.71 + b[y, x] * 0.07
__global__ void toGrayKernel(
    const unsigned char* I,
    unsigned char* O,
    size_t w,
    size_t h
) {
    // Remember that this is a 2D grid call
    // So which column (or x) are we working at? 
    size_t col = blockDim.x * blockIdx.x + threadIdx.x;
    // Similarly the row we are working at
    size_t row = blockDim.y * blockIdx.y + threadIdx.y;

    // Bail if we have a column/ row greater than the width and height of the image
    if ( row >= h || col >= w) {
        return;
    }

    // We have read the image in `row-major` order - so our target pixel is row * number of columns/ pixels in a row + column we are at
    size_t grayat = row * w + col;
    // Now, since our RGB image is 3 channels, the `r` is at target pixel * 3, `g` is at target (pixel * 3) + 1 .. and so on
    size_t rgb = grayat * 3;

    // Finally compute the value and put the target pixel
    O[grayat] = I[rgb] * 0.21f + I[rgb + 1] * 0.71f + I[rgb + 2] * 0.07f;
}

void toGray2D(
    const unsigned char* I_h,
    unsigned char* O_h,
    size_t w,
    size_t h) {
        // declare the device input and output
        // using `_d` convention to point that these are going to be pointing to device memory
        unsigned char *I_d, *O_d;

        // The dimensions of the image
        size_t imdim = w * h;
        // Output is single channel
        size_t outsize = imdim * sizeof(unsigned char);
        // Input is 3 channels
        size_t imsize = outsize * 3;

        // Allocating device memory for input and output
        cudaMalloc((void**)&I_d, imsize);
        cudaMalloc((void**)&O_d, outsize);

        // Copying input buffer to device
        cudaMemcpy(I_d, I_h, imsize, cudaMemcpyHostToDevice);

        // Gets interesting here - we are using a 2d grid blocks
        // Max number of threads per block is capped at 1024 - so we are defining 32 * 32 * 1 size of threads in block
        dim3 threadsPerBlock = dim3(32, 32, 1);
        // Since we are capping the threads per block along x at 32 - we are saying that our max blocks x would be ceil(w / 32)
        // Again, since we have 32 threads along y dim of a block we are capping the num blocks across y dim at ceil(h / 32)
        dim3 numBlocks = dim3(ceil(w / 32), ceil(h / 32), 1);

        // Calling the kernel with these settings
        toGrayKernel<<< numBlocks, threadsPerBlock>>>(I_d, O_d, w, h);

        // Copying the device output result back to host output
        cudaMemcpy(O_h, O_d, outsize, cudaMemcpyDeviceToHost);

        // Freeing the cuda allocations to the pool
        cudaFree(I_d);
        cudaFree(O_d);
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