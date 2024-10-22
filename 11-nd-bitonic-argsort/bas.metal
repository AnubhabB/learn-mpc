R"(#include <metal_stdlib>
using namespace metal;

#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }
#define SORT_ASC 1
#define SORT_DESC 0

kernel void bitonicArgSort(
    device const float    * x,
    device       int * dst,
    constant     int64_t  & ncols,
    constant     int64_t  & ncols_pad,
    // threadgroup  uint32_t * shared_values [[threadgroup(0)]],
    uint2 tgpig[[threadgroup_position_in_grid]],
    uint2 tpitg[[thread_position_in_threadgroup]]
) {
    int order = 0;
    int col = tpitg[0] + tgpig[0] * 1024;
    int row = tgpig[1];

    if (col >= ncols_pad) return;

    device const float    * x_row = x + row * ncols;
    device int    * dst_row       = dst;
    // threadgroup int  * dst_row = shared_values;

    // initialize indices
    // dst_row[col] = 1;
    dst_row[col] = col;

    threadgroup_barrier(mem_flags::mem_none);
    
    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (dst_row[col] >= ncols ||
                        (dst_row[ixj] < ncols && (order == SORT_ASC ?
                            x_row[dst_row[col]] > x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] < x_row[dst_row[ixj]]))
                    ) {
                        SWAP(dst_row[col], dst_row[ixj]);
                    }
                } else {
                    if (dst_row[ixj] >= ncols ||
                        (dst_row[col] < ncols && (order == SORT_ASC ?
                            x_row[dst_row[col]] < x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] > x_row[dst_row[ixj]]))
                    ) {
                        SWAP(dst_row[col], dst_row[ixj]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_none);
        }
    }
    

    // copy the result to dst without the padding
    if (col < ncols) {
        //dst[row * ncols + col] = dst_row[col];
        dst[row * ncols + col] = dst_row[col];
    }
})"