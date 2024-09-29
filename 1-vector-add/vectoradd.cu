// The `__global__` denotes that this is a `kernel` function that can be executed from the host (or even device)
// Threads in CUDA are arranges in a 2-level hierarchy
// - a group of threads form a `block`
// - a group of blocks form a `grid`
// Each thread has a unique `id` based on the index - current indexing pattern is 1D
__global__ void vecAddKernel(const float *A, const float *B, float *C, int N) {
    // E.g.
    // 0th block, first thread would have index `0`
    // 0th block, second thread will have index `1`
    // 2nd block, first thread would have `1024 * 1 + 0 => index 1024` 
    // 3nd block, fifth thread would have index `1024 * 2 + 4 => 2052`
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// vecAdd receives
// A, B, C are pointers to float vectors - `_h` is used to denote that these reside in the `host` memory
// A & B are the input data and C will hold the addition output
void vecAdd(float *A_h, float *B_h, float *C_h, int n) {
    // Calculate the size in bits of the vector -
    // in this case since this is a `single precision float` we are effectively saying `n * 4 bytes` or `n * 32 bits`
    int size = n * sizeof(float);
    // declaring the variables - A and B will hold the input vector
    // C will hold the output
    // the `_d` suffix is to just say that these variables are going to be residing in `device` memory
    // Dereferencing `device` memory variables in `host` may lead to panics and errors
    float *A_d, *B_d, *C_d;

    // using cuda API to allocate the required memory locations in the device `global` memory
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);

    // Copy the inputs from source to destination
    // In our case source is the vector in `host`
    // Destination is the `device`
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // Kernel call
    // The triple shevron syntax `<<<Number_of_blocks, Threads_per_block>>>`
    // The max threads per blocks supported by modern systems is upto 1024
    // We are basically calculating the blocks to launch based simply on the total number of elements that needs to be processed
    vecAddKernel<<<ceil(n/ 1024.), 1024>>>(A_d, B_d, C_d, n);

    // Copy the results back to the host memory
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
    // finally freeing up the device memory - release allocated device memory to the memory pool
    cudaFree(&A_d);
    cudaFree(&B_d);
    cudaFree(&C_d);
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
        if ( C[i] != 5. ) {
            return 1;
        }
    }

    free(A);
    free(B);
    free(C);

    return 0;
}