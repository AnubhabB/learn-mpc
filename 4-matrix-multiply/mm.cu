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
    size_t col = blockDim.x * blockIdx.x + threadIdx.x;
    size_t row = blockDim.y * blockIdx.y + threadIdx.y;
    if (col >= P || row >= M) {
        return;
    }

    float v = 0.;
    for(int i=0; i<N; i++) {
        size_t a_item = row * N + i;
        size_t b_item = col + P * i;
        v += A[a_item] * B[b_item];
    }

    C[row * P + col] = v;
}

int main() {
    const size_t M = 2048;
    const size_t N = 4096;
    const size_t P = 2048;

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

    // kernel call
    dim3 threadsPerBlock(32, 32, 1);
    dim3 numBlocks(static_cast<size_t>(ceil((static_cast<float>(P) / threadsPerBlock.x))),
        static_cast<size_t>(ceil((static_cast<float>(M) / threadsPerBlock.y))));

    naive<<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d, M, N, P);


    cudaMemcpy(C, C_d, M * P * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    free(A);
    free(B);
    free(C);

    return 0;
}