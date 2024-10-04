#include "../imread.h"

__global__ void naiveBlurKernel(
    const unsigned char* I,
    unsigned char* O,
    const int RADIUS,
    const size_t w,
    const size_t h,
    const size_t chan
) {
    size_t col = blockDim.x * blockIdx.x + threadIdx.x;
    size_t row = blockDim.y * blockIdx.y + threadIdx.y;
    
    if ( !(row < h && col < w)) {
        return;
    }


    float r = 0.;
    float g = 0.;
    float b = 0.; 

    float px_count = 0.;

    // (col: 1, row: 1)
    // (row * w + col, row * w + col + 1, row * w + col + 2)
    for(int blurRow = -RADIUS; blurRow < RADIUS + 1; blurRow++) {
        for(int blurCol = -RADIUS; blurCol < RADIUS + 1; blurCol++) {
            int curRow = row + blurRow;
            int curCol = col + blurCol;

            if(curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
                size_t at = (curRow * w + curCol) * chan;
                for(int i=0; i<chan; i++) {
                    if(i == 0) {
                        r += (float) I[at];
                    } else if(i == 1) {
                        g += (float) I[at + 1];
                    } else if(i == 2) {
                        b += (float) I[at + 2];
                    }
                }
                
                ++px_count;
            }
        }
    }

    size_t trg = (row * w + col) * chan;
    for(int i=0; i<chan; i++) {
        int v;
        if(i == 0) {
            v = r;
        } else if(i == 1) {
            v = g;
        } else if(i == 2) {
            v = b;
        }

        O[trg + i] = (unsigned char) round(v / px_count);
    }
}

void naiveBlur(
    const unsigned char* I_h,
    unsigned char* O_h,
    size_t w,
    size_t h,
    const int radius,
    size_t chan
) {
        cudaError_t e;
        // declare the device input and output
        // using `_d` convention to point that these are going to be pointing to device memory
        unsigned char *I_d, *O_d;

        // The dimensions of the image
        // Output is single channel
        size_t size = w * h * sizeof(unsigned char) * chan;

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
        dim3 threadsPerBlock = dim3(32, 32, 1);
        // Since we are capping the threads per block along x at 32 - we are saying that our max blocks x would be ceil(w / 32)
        // Again, since we have 32 threads along y dim of a block we are capping the num blocks across y dim at ceil(h / 32)
        dim3 numBlocks = dim3(ceil(w / 32), ceil(h / 32), 1);

        // Calling the kernel with these settings
        naiveBlurKernel<<< numBlocks, threadsPerBlock>>>(I_d, O_d, radius, w, h, chan);
        
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
    unsigned char * img, * out;
    size_t w, h, size, chan;

    // Read image with a helper function
    tie(img, w, h, chan) = imRead("data/gray-lion.jpg");

    size = w * h;
    // Allocate data to the output based on the size of the image. Single channel so (w * h * 1)
    out = (unsigned char*)malloc(size * chan * sizeof(unsigned char));
    // call the kernel caller
    naiveBlur(img, out, w, h, RADIUS, chan);
    // helper function to save the result
    jpeg_write(out, w, h, chan, "data/blur-lion-grey.jpg");
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
    jpeg_write(out, w, h, chan, "data/blur-lion.jpg");
    // Free the allocations
    free(img);
    free(out);
    return 0;
}