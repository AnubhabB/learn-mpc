R"(#include <metal_stdlib>
using namespace metal;

kernel void naive(
    device const float *A,
    device const float *B,
    device float *C,
    constant size_t &M,
    constant size_t &N,
    constant size_t &P,
    uint2 index[[thread_position_in_grid]]
) {
        // Remember that this is a 2D grid call
        // So which column (or x) are we working at? 
        size_t col = index.x;
        size_t row = index.y;

        if (col >= P || row >= M) {
            return;
        }

        float v = 0.;

        for(size_t i=0; i<N; i++) {
            size_t a_idx = row * N + i;
            size_t b_idx = col + P * i;

            v += A[a_idx] * B[b_idx];
        }

        C[row * P + col] = v;
})"