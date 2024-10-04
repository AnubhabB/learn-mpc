R"(#include <metal_stdlib>
using namespace metal;

kernel void to_blur_kernel(
    device const unsigned char *I,
    device unsigned char *O,
    constant size_t &w,
    constant size_t &h,
    constant int &radius,
    constant unsigned char &chan,
    uint2 index[[thread_position_in_grid]]) {
        // Remember that this is a 2D grid call
        // So which column (or x) are we working at? 
        size_t col = index.x;
        size_t row = index.y;

        if (row >= h || col >= w) {
            return;
        }

        float r = 0.;
        float g = 0.;
        float b = 0.; 

        float px_count = 0.;

        for(int blurRow = -radius; blurRow < radius + 1; blurRow++) {
            for(int blurCol = -radius; blurCol < radius + 1; blurCol++) {
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
})"