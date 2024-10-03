R"(#include <metal_stdlib>
using namespace metal;

kernel void vec_add(
    device const float* A,
    device const float* B,
    device float* C,
    constant size_t &n,
    uint index[[thread_position_in_grid]]) {
        if(index >= n) {
            return;
        }

        C[index] = A[index] + B[index];
})"