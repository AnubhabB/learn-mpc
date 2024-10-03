R"(#include <metal_stdlib>
using namespace metal;

kernel void to_gray_kernel(
    device const unsigned char *I,
    device unsigned char *O,
    constant size_t &w,
    constant size_t &h,
    uint2 index[[thread_position_in_grid]]) {
        // Remember that this is a 2D grid call
        // So which column (or x) are we working at? 
        size_t col = index.x;
        size_t row = index.y;

        if (row >= h || col >= w) {
            return;
        }

        // We have read the image in `row-major` order - so our target pixel is row * number of columns/ pixels in a row + column we are at
        size_t grayat = row * w + col;
        // Now, since our RGB image is 3 channels, the `r` is at target pixel * 3, `g` is at target (pixel * 3) + 1 .. and so on
        size_t rgb = grayat * 3;

        // Finally compute the value and put the target pixel
        O[grayat] = I[rgb] * 0.21f + I[rgb + 1] * 0.71f + I[rgb + 2] * 0.07f;
})"