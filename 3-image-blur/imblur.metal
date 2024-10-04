R"(#include <metal_stdlib>
using namespace metal;

kernel void to_blur_kernel(
    device const unsigned char *I,
    device unsigned char *O,
    device const size_t &w,
    device const size_t &h,
    device const int &radius,
    device const unsigned char &chan,
    uint2 index[[thread_position_in_grid]]) {
        // Remember that this is a 2D grid call
        // So which column (or x) are we working at? 
        size_t col = index.x;
        size_t row = index.y;

        if (row >= h || col >= w) {
            return;
        }

        unsigned int px_val = 0;
        float px_count = 0.;

        for(int blurRow = -radius; blurRow < radius + 1; blurRow++) {
            for(int blurCol = -radius * chan; blurCol < radius * chan + 1; blurCol += chan) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                if(curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
                    size_t at = (curRow * w + curCol);
                    px_val += static_cast<unsigned int>(I[at]);
                    ++px_count;
                }
            }
        }

        O[(row * w + col)] = static_cast<unsigned char>(round(px_val / px_count));
})"