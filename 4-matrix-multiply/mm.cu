#include <iostream>

// Multiply a MxN . NxP matrix
/*
[1, 2, 3] . [1, 2] = [22,  28]
[4, 5, 6]   [3, 4]   [49,  64]
[7, 8, 9]   [5, 6]   [76, 100]
*/
__global__ void naive(
    const float *A,
    const float *B,
    float *C,
    size_t M,
    size_t N,
    size_t P
) {
    size_t col = blockDim.x * blockIdx.x + threadIdx.x; // M in sample
    size_t row = blockDim.y * blockIdx.y + threadIdx.y; // N in sample
    if (col >= P || row >= M) {
        return;
    }

    float v = 0.;
    for(int i=0; i<N; i++) { // K in sample
        size_t a_item = row * N + i;
        size_t b_item = col + P * i;
        v += A[a_item] * B[b_item];
    }

    C[row * P + col] = v;
}

// https://siboehm.com/articles/22/CUDA-MMM - Kernel 2: Global Memory Coalescing
template <const uint BLOCKSIZE>
__global__ void coalesc(
    const float *A,
    const float *B,
    float *C,
    size_t M,
    size_t N,
    size_t P
) {
    size_t col = blockIdx.x * BLOCKSIZE + ( threadIdx.x / BLOCKSIZE); // M in sample
    size_t row = blockIdx.y * BLOCKSIZE + ( threadIdx.y % BLOCKSIZE); // N in sample
    if (col >= P || row >= M) {
        return;
    }

    float v = 0.;
    for(int i=0; i<N; i++) { // K in sample
        size_t a_item = row * N + i;
        size_t b_item = col + P * i;
        v += A[a_item] * B[b_item];
    }

    C[row * P + col] = v;
}

template <const uint BLOCKSIZE>
__global__ void shared_mem(const float *A,
    const float *B,
    float *C,
    size_t M,
    size_t N,
    size_t P
) {
    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    __shared__ float A_s[BLOCKSIZE * BLOCKSIZE];
    __shared__ float B_s[BLOCKSIZE * BLOCKSIZE];

    const uint threadCol = threadIdx.x % BLOCKSIZE;
    const uint threadRow = threadIdx.x / BLOCKSIZE;
    
    A += cRow * BLOCKSIZE * N;                    // row=cRow, col=0
    B += cCol * BLOCKSIZE;                        // row=0, col=cCol
    C += cRow * BLOCKSIZE * P + cCol * BLOCKSIZE; // row=cRow, col=cCol

    float tmp = 0.0;
    for (int bkIdx = 0; bkIdx < N; bkIdx += BLOCKSIZE) {
        // Have each thread load one of the elements in A & B
        // Make the threadCol (=threadIdx.x) the consecutive index
        // to allow global memory access coalescing
        A_s[threadRow * BLOCKSIZE + threadCol] = A[threadRow * N + threadCol];
        B_s[threadRow * BLOCKSIZE + threadCol] = B[threadRow * P + threadCol];
        
        // block threads in this block until cache is fully populated
        __syncthreads();
        A += BLOCKSIZE;
        B += BLOCKSIZE * P;
        
        // execute the dotproduct on the currently cached block
        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
            tmp += A_s[threadRow * BLOCKSIZE + dotIdx] *
                B_s[dotIdx * BLOCKSIZE + threadCol];
        }
        
        // need to sync again at the end, to avoid faster threads
        // fetching the next block into the cache before slower threads are done
        __syncthreads();
    }

    C[threadRow * P + threadCol] = tmp;
}

void CudaDeviceInfo() {
  int deviceId;

  cudaGetDevice(&deviceId);

  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, deviceId);

  printf("Device ID: %d\n\
    Name: %s\n\
    Compute Capability: %d.%d\n\
    memoryBusWidth: %d\n\
    maxThreadsPerBlock: %d\n\
    maxThreadsPerMultiProcessor: %d\n\
    maxRegsPerBlock: %d\n\
    maxRegsPerMultiProcessor: %d\n\
    totalGlobalMem: %zuMB\n\
    sharedMemPerBlock: %zuKB\n\
    sharedMemPerMultiprocessor: %zuKB\n\
    totalConstMem: %zuKB\n\
    multiProcessorCount: %d\n\
    Warp Size: %d\n",
         deviceId, props.name, props.major, props.minor, props.memoryBusWidth,
         props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor,
         props.regsPerBlock, props.regsPerMultiprocessor,
         props.totalGlobalMem / 1024 / 1024, props.sharedMemPerBlock / 1024,
         props.sharedMemPerMultiprocessor / 1024, props.totalConstMem / 1024,
         props.multiProcessorCount, props.warpSize);
};

int main() {
    CudaDeviceInfo();

    const size_t M = 4096;
    const size_t N = 4096;
    const size_t P = 4096;
    // const size_t M = 4;
    // const size_t N = 3;
    // const size_t P = 2;

    float *A_d, *B_d, *C_d;

    float *A = (float*)malloc(M * N * sizeof(float));
    float *B = (float*)malloc(N * P * sizeof(float));
    float *C = (float*)malloc(M * P * sizeof(float));

    for(int i = 0; i < M * N; i++) {
        A[i] = static_cast<float>(i);
        if( i < N * P) {
            B[i] = static_cast<float>(i);
        }
    }

    cudaMalloc((void**)&A_d, M * N * sizeof(float));
    cudaMalloc((void**)&B_d, N * P * sizeof(float));
    cudaMalloc((void**)&C_d, M * P * sizeof(float));

    cudaMemcpy(A_d, A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, N * P * sizeof(float), cudaMemcpyHostToDevice);

    // kernel call - naive
    dim3 threadsPerBlockNaive(32, 32, 1);
    dim3 numBlocksNaive(static_cast<size_t>(ceil((static_cast<float>(P) / threadsPerBlockNaive.x))),
        static_cast<size_t>(ceil((static_cast<float>(M) / threadsPerBlockNaive.y))));

    naive<<<numBlocksNaive, threadsPerBlockNaive>>>(A_d, B_d, C_d, M, N, P);

    cudaMemcpy(C, C_d, M * P * sizeof(float), cudaMemcpyDeviceToHost);

    float *baseline = (float*)malloc(M * P * sizeof(float));
    
    memcpy(baseline, C, M * P * sizeof(float));
    free(C);

    C = (float*)malloc(M * P * sizeof(float));

    // kernel call - coalesced
    dim3 threadsPerBlockCoalesc(32 * 32);
    dim3 numBlocksCoalesc(static_cast<size_t>(ceil((static_cast<float>(P) / 32))),
        static_cast<size_t>(ceil((static_cast<float>(M) / 32))));

    coalesc<384><<<numBlocksCoalesc, threadsPerBlockCoalesc>>>(A_d, B_d, C_d, M, N, P);

    cudaMemcpy(C, C_d, M * P * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i=0; i < M*P; i++) {
        if(baseline[i] != C[i]) {
            printf("Error coalesced @index[%d]: %f %f", i, baseline[i], C[i]);
            break;
        }
    }

    free(C);
    C = (float*)malloc(M * P * sizeof(float));

    dim3 threadsPerBlockSharedMem(32 * 32);
    dim3 numBlocksSharedMem(static_cast<size_t>(ceil((static_cast<float>(M) / 32))),
        static_cast<size_t>(ceil((static_cast<float>(P) / 32))));

    shared_mem<32><<<numBlocksSharedMem, threadsPerBlockSharedMem>>>(A_d, B_d, C_d, M, N, P);

    cudaMemcpy(C, C_d, M * P * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i=0; i < M*P; i++) {
        if(baseline[i] != C[i]) {
            printf("Error shared mem @index[%d]: %f %f", i, baseline[i], C[i]);
            break;
        }
    }

    free(C);
    C = (float*)malloc(M * P * sizeof(float));

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    free(A);
    free(B);
    free(C);

    return 0;
}