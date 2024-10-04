#include "../imread.h"

__global__ void naiveBlurKernel(
    const unsigned char* I,
    unsigned char* O,
    const size_t w,
    const size_t h,
    const int radius,
    const unsigned char chan
) {
    size_t col = blockDim.x * blockIdx.x + threadIdx.x;
    size_t row = blockDim.y * blockIdx.y + threadIdx.y;
    
    if ( !(row < h && col < w)) {
        return;
    }

    unsigned int px_val = 0;
    float px_count = 0.;

    // (col: 1, row: 1)
    // (row * w + col, row * w + col + 1, row * w + col + 2)
    for(int blurRow = -radius; blurRow < radius + 1; blurRow++) {
        for(int blurCol = -radius * chan; blurCol < radius * chan + 1; blurCol += chan) {
            int curRow = row + blurRow;
            int curCol = col + blurCol;

            if(curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
                size_t at = curRow * w + curCol;
                px_val += static_cast<unsigned int>(I[at]);
                ++px_count;
            }
        }
    }

    O[(row * w + col)] = static_cast<unsigned char>(round(px_val / px_count));
}

void naiveBlur(
    const unsigned char* I_h,
    unsigned char* O_h,
    size_t w,
    size_t h,
    const int radius,
    const unsigned char chan
) {
        cudaError_t e;
        // declare the device input and output
        // using `_d` convention to point that these are going to be pointing to device memory
        unsigned char *I_d, *O_d;

        // The dimensions of the image
        // Output is single channel
        size_t true_w = w * chan;
        size_t size = true_w * h * sizeof(unsigned char);

        // Allocating device memory for input and output
        e = cudaMalloc((void**)&I_d, size);
        if(e) {
            printf("%s", cudaGetErrorString(e));
            return;
        }
        e = cudaMalloc((void**)&O_d, size);
        if(e) {
            printf("%s", cudaGetErrorString(e));
            return;
        }

        // Copying input buffer to device
        e = cudaMemcpy(I_d, I_h, size, cudaMemcpyHostToDevice);
        if(e) {
            printf("%s", cudaGetErrorString(e));
            return;
        }

        // Gets interesting here - we are using a 2d grid blocks
        // Max number of threads per block is capped at 1024 - so we are defining 32 * 32 * 1 size of threads in block
        dim3 threadsPerBlock(32, 32, 1);
        // Since we are capping the threads per block along x at 32 - we are saying that our max blocks x would be ceil(w / 32)
        // Again, since we have 32 threads along y dim of a block we are capping the num blocks across y dim at ceil(h / 32)
        dim3 numBlocks(static_cast<size_t>(ceil(static_cast<float>(true_w) / threadsPerBlock.x)),
               static_cast<size_t>(ceil(static_cast<float>(h) / threadsPerBlock.y)));
        // dim3 numBlocks = dim3(ceil(w / 32), ceil(h / 32), 1);

        // Calling the kernel with these settings
        naiveBlurKernel<<<numBlocks, threadsPerBlock>>>(I_d, O_d, true_w, h, radius, chan);
        
        // Copying the device output result back to host output
        e = cudaMemcpy(O_h, O_d, size, cudaMemcpyDeviceToHost);
        if(e) {
            printf("%s", cudaGetErrorString(e));
            return;
        }

        // Freeing the cuda allocations to the pool
        cudaFree(I_d);
        cudaFree(O_d);
}

int main() {
    const int RADIUS = 64;
    // Allocate buffers to input and output image. Input image is 3 channels while output is 1
    unsigned char * img, * out, chan;
    size_t w, h, size;

    // Read image with a helper function
    tie(img, w, h, chan) = imRead("data/gray-lion.jpg");

    size = w * h;
    // Allocate data to the output based on the size of the image. Single channel so (w * h * 1)
    out = (unsigned char*)malloc(size * chan * sizeof(unsigned char));
    // call the kernel caller
    naiveBlur(img, out, w, h, RADIUS, chan);
    // helper function to save the result
    jpeg_write(out, w, h, chan, "data/blur-lion-grey-c.jpg");
    // Free the allocations
    free(img);
    free(out);

    // Read image with a helper function
    tie(img, w, h, chan) = imRead("data/lion.jpg");

    size = w * h;
    // Allocate data to the output based on the size of the image. Single channel so (w * h * 1)
    out = (unsigned char*)malloc(size * chan * sizeof(unsigned char));
    // call the kernel caller
    naiveBlur(img, out, w, h, RADIUS, chan);
    // helper function to save the result
    jpeg_write(out, w, h, chan, "data/blur-lion-c.jpg");
    // Free the allocations
    free(img);
    free(out);
    return 0;
}