#include <stdio.h>
#include <stdint.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define N_THREADS           256
#define RADIX               256             //Number of digit bins
#define WARP_SIZE           32
#define LANE_LOG            5               // LANE_LOG = 5 since 2^5 = 32 = warp size
#define RADIX_LOG           8               // 2^8 = 258

#define LANE_MASK           (WARP_SIZE - 1)
#define RADIX_MASK          (RADIX - 1)     //Mask of digit bins, to extract digits
#define WARP_INDEX          (threadIdx.x >> LANE_LOG)

#define BIN_KEYS_PER_THREAD 15
#define SUB_PARTITION_SIZE (BIN_KEYS_PER_THREAD * WARP_SIZE);

// Thread position within a warp
__device__ __forceinline__ uint32_t getLaneId() {
    uint32_t laneId;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
    return laneId;
}

// Gets a bit mask representing all lanes with IDs less than the current thread's lane ID within a warp
__device__ __forceinline__ unsigned getLaneMaskLt() {
    unsigned mask;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
    return mask;
}

// Prefix sum
// Performs an inclusive scan operation within a single warp
// Uses butterfly/sequential addressing pattern for efficiency
__device__ __forceinline__ uint32_t InclusiveWarpScan(uint32_t val) {
    #pragma unroll
    for (int offset = 1; offset <= 16; offset <<= 1) {
        const uint32_t t = __shfl_up_sync(0xffffffff, val, offset);
        if (getLaneId() >= offset) val += t;
    }

    return val;
}

// Circular shift prefix sum
__device__ __forceinline__ uint32_t InclusiveWarpScanCircularShift(uint32_t val) {
    #pragma unroll
    for (int offset = 1; offset <= 16; offset <<= 1) {
        const uint32_t t = __shfl_up_sync(0xffffffff, val, offset);
        if (getLaneId() >= offset) val += t;
    }

    return __shfl_sync(0xffffffff, val, getLaneId() + LANE_MASK & LANE_MASK);
}

// Helper for active warp scan (used for inter-warp scan) with early termination
__device__ __forceinline__ uint32_t ActiveInclusiveWarpScan(uint32_t val) {
    const uint32_t mask = __activemask();
    const int active_threads = __popc(mask);

    #pragma unroll
    for (int offset = 1; offset <= 16; offset <<= 1) {
        if (offset >= active_threads) break;  // Early termination
        const uint32_t t = __shfl_up_sync(mask, val, offset);
        if (getLaneId() >= offset) val += t;
    }

    return val;
}

// Prefix sum of active threads ot excluding itself
__device__ __forceinline__ uint32_t ActiveExclusiveWarpScan(uint32_t val) {
    const uint32_t mask = __activemask();
    #pragma unroll
    for (int offset = 1; offset <= 16; offset <<= 1) {
        const uint32_t t = __shfl_up_sync(mask, val, offset);
        if (getLaneId() >= offset) val += t;
    }

    const uint32_t t = __shfl_up_sync(mask, val, 1);
    return getLaneId() ? t : 0;
}

// Helper functions for bit conversions
template<typename T>
__device__ inline uint32_t toBits(T val) {
    if constexpr (std::is_same<T, float>::value) {
        return __float_as_uint(val) ^ ((__float_as_uint(val) >> 31) | 0x80000000);
    }
    else if constexpr (std::is_same<T, __half>::value) {
        uint16_t bits = __half_as_ushort(val);
        uint16_t mask = -int(bits >> 15) | 0x8000;
        return static_cast<uint32_t>(bits ^ mask);
    }
    else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        uint16_t bits = __bfloat16_as_ushort(val);
        uint16_t mask = -int(bits >> 15) | 0x8000;
        return static_cast<uint32_t>(bits ^ mask);
    }
    else if constexpr (std::is_same<T, int64_t>::value) {
        // return static_cast<uint32_t>((val >> radixShift) & 0xFFFFFFFF);
        // TODO - how to handle int64?????
    }
    else {
        return static_cast<uint32_t>(val);
    }
}

// Vector type mappings
template<typename T> struct VectorTrait {
    // Default case (32-bit types)
    static constexpr uint32_t vector_size = 4;  // 4 * 4 bytes = 16 bytes
    static constexpr uint32_t bytes_per_vector = sizeof(T) * vector_size;
};

// Vectorizations for different types
template<>
struct VectorTrait<half> {
    static constexpr uint32_t vector_size = 8;  // 8 * 2 bytes = 16 bytes
    // using type = float4;
};

template<>
struct VectorTrait<nv_bfloat16> {
    static constexpr uint32_t vector_size = 8;   // 8 * 2 bytes = 16 bytes
};

// Specialization for 8-bit types
template<>
struct VectorTrait<uint8_t> {
    static constexpr uint32_t vector_size = 16;  // 16 * 1 byte = 16 bytes
};


// Helper function to get type-specific maximum value
template<typename T>
__device__ inline T getTypeMax() {
    if constexpr (std::is_same<T, float>::value) {
        return INFINITY;
    }
    else if constexpr (std::is_same<T, __half>::value) {
        return __float2half(INFINITY);
    }
    else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        return __float2bfloat16(INFINITY);
    }
    else if constexpr (std::is_same<T, int64_t>::value) {
        return 0x7FFFFFFFFFFFFFFF;
    } else if constexpr (std::is_same<T, unsigned char>::value) {
        return 0xFF; // 255 in hex
    } else if constexpr (std::is_same<T, u_int32_t>::value) {
        return 0xFFFFFFFF;  // 4294967295 in hex
    } else {
        // This seems to be experimental
        // calling a constexpr __host__ function("max") from a __device__ function("getTypeMax") is not allowed. The experimental flag '--expt-relaxed-constexpr' can be used to allow this.
        
        // Shouldn't reach here
        return static_cast<T>(-1);
    }
}

// Radix Upsweep pass does the following:
// radixShift - signifies which `digit` position is being worked on in strides of 8 - first pass for MSB -> last 8 bits using radix 256
// passHist - for a particular digit position creates a frequency of values -
// in this implementaiton a passHist is computer per threadBlock and each threadBlock is responsible for processing `numElementsInBlock`
// globalHist - converts these frequencies into cumulative counts (prefix sums)
template<typename T>
__global__ void RadixUpsweep(
    T* sort,
    uint32_t* globalHist,
    uint32_t* passHist,
    const uint32_t size,
    const uint32_t radixShift,
    const uint32_t maxElemInBlock // max number of elements being processed by this block
) {
    uint32_t printBlock = 0;
    // Shared memory for histogram - two sections to avoid bank conflicts
    constexpr uint32_t sharedSize = RADIX * 2;
    __shared__ uint32_t s_globalHist[sharedSize];

    // Clear shared memory histogram
    #pragma unroll
    for (uint32_t i = threadIdx.x; i < sharedSize; i += blockDim.x)
        s_globalHist[i] = 0;
    __syncthreads();

    // Calculate this block's range
    const uint32_t block_start = blockIdx.x * maxElemInBlock;
    const uint32_t block_end = min(block_start + maxElemInBlock, size);
    const uint32_t elements_in_block = block_end - block_start;

    // if(blockIdx.x == printBlock && (threadIdx.x == 0 || threadIdx.x == 15)) {
    //     printf("Thread[%u]: maxElemInBlock[%u] block_start[%u] block_end[%u] elements_in_block[%u]\n", threadIdx.x, maxElemInBlock, block_start, block_end, elements_in_block);
    // }

    // if(threadIdx.x == 0 && blockIdx.x == printBlock) {
    //     // printf("\nBlock[%u]: ", blockIdx.x);
    //     for(uint32_t i=block_start; i<block_end; ++i) {
    //         printf("%u ", sort[i]);
    //     }
    //     printf("--\n");
    // }
    // Vector load based on types
    constexpr uint32_t vec_size = VectorTrait<T>::vector_size;

    // Calculate number of full vectors - we are going to make an attempt to process
    const uint32_t full_vecs = elements_in_block / vec_size;
    const uint32_t vec_end = block_start + (full_vecs * vec_size);
    
    for (uint32_t i = threadIdx.x; i < full_vecs; i += blockDim.x) {
        const uint32_t idx = block_start + i * vec_size;
        
        if(idx < vec_end) {
            // if(blockIdx.x == printBlock) {
            //     printf("Idx[%u] ", idx);
            // }
            #pragma unroll
            for (int j = 0; j < vec_size; ++j) {
                uint32_t bits = toBits(sort[idx + j]);
                // if(blockIdx.x == printBlock) {
                //     printf("sort[%u %u]: [%u %u %u] ", idx, idx + j, sort[idx + j], bits, bits >> radixShift & RADIX_MASK);
                // }
                atomicAdd(&s_globalHist[bits >> radixShift & RADIX_MASK], 1);
            }
        }
    }
    
    // Process remaining elements
    for (uint32_t i = threadIdx.x + vec_end; i < block_end; i += blockDim.x) {
        uint32_t bits = toBits(sort[i]);
        atomicAdd(&s_globalHist[bits >> radixShift & RADIX_MASK], 1);
    }

    __syncthreads();

    // Reduce histograms and prepare for prefix sum
    for (uint32_t i = threadIdx.x; i < RADIX; i += blockDim.x) {
        // Merge possible bank conflicts
        s_globalHist[i] += s_globalHist[i + RADIX];
        // Memory layout: digit frequencies across all blocks
        // So, if we have n blocks we'll have frequency values for a digit in each blocks consecutively
        passHist[i * gridDim.x + blockIdx.x] = s_globalHist[i];
        s_globalHist[i] = InclusiveWarpScanCircularShift(s_globalHist[i]);
    }
    __syncthreads();
    // if(threadIdx.x == 15 && blockIdx.x == printBlock) {
    //     printf("\nBlockHist[%u]: ", blockIdx.x);
    //     for(uint32_t i=0; i<RADIX; ++i) {
    //         printf("[%u %u %u]\n", i, passHist[RADIX * blockIdx.x + i], s_globalHist[i]);
    //     }
    //     printf("--\n");
    // }

    // Perform warp-level scan - for first thread in each warp
    if (threadIdx.x < (RADIX >> LANE_LOG))
        s_globalHist[threadIdx.x << LANE_LOG] = ActiveExclusiveWarpScan(s_globalHist[threadIdx.x << LANE_LOG]);
    __syncthreads();

    // if(threadIdx.x == 15 && blockIdx.x == printBlock) {
    //     printf("\nBlockHist[%u]: ", blockIdx.x);
    //     for(uint32_t i=0; i<RADIX; ++i) {
    //         printf("[%u %u %u]\n", i, passHist[RADIX * blockIdx.x + i], s_globalHist[i]);
    //     }
    //     printf("--\n");
    // }
    // Update global histogram with prefix sum results
    for (uint32_t i = threadIdx.x; i < RADIX; i += blockDim.x) {
        atomicAdd(
            &globalHist[i + (radixShift << LANE_LOG)], 
            s_globalHist[i] + 
            (getLaneId() ? 
                __shfl_sync(0xfffffffe, s_globalHist[i - 1], 1) : 0)
        );
    }

    // if(threadIdx.x == 15 && blockIdx.x == printBlock) {
    //     printf("\nBlockHist[%u]: ", blockIdx.x);
    //     for(uint32_t i=0; i<RADIX; ++i) {
    //         printf("[%u %u %u]\n", i, passHist[RADIX * blockIdx.x + i], s_globalHist[i]);
    //     }
    //     printf("--\n");
    // }
}


__global__ void RadixScan(
    uint32_t* passHist,
    const uint32_t numBlocks
) {
    const uint32_t blockSize = blockDim.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t laneId = getLaneId();

    __shared__ uint32_t s_scan[128];

    // Initialize the shared memory!
    s_scan[tid] = 0;
    __syncthreads();

    
    // Circular shift within warp - this helps reduce bank conflicts
    // Get ID of the next thread: getLaneId(): 0 -> 1, 1 -> 2 ... 31 -> 0
    const uint32_t circularLaneShift = (laneId + 1) & LANE_MASK;
    
    if(threadIdx.x == 24 && blockIdx.x == 4) {
        printf("Lane: %u %u\n", circularLaneShift, laneId);
    }

    // Where does the digit start
    const uint32_t digitOffset = blockIdx.x * numBlocks;

    // Calculate the number of full block-sized chunks we need to process
    const uint32_t fullBlocksEnd = (numBlocks / blockSize) * blockSize;
    
    // Running sum for carrying over between iterations
    uint32_t reduction = 0;

    // Process full blocks
    // for (uint32_t blockStart = 0; blockStart < fullBlocksEnd; blockStart += blockSize) {
    //     // Load data into shared memory with circular shift pattern
    //     const uint32_t globalIdx = blockStart + tid;
    //     s_scan[tid] = passHist[globalIdx + digitOffset];

    //     // Step 1: Perform warp-level scan
    //     s_scan[tid] = InclusiveWarpScan(s_scan[tid]);
    //     __syncthreads();

    //     // Step 2: Collect and scan warp totals
    //     if (tid < (blockDim.x >> LANE_LOG)) {
    //         s_scan[((tid + 1) << LANE_LOG) - 1] = ActiveInclusiveWarpScan(s_scan[((tid + 1) << LANE_LOG) - 1]);
    //     }
    //     __syncthreads();

    //     const uint32_t writeIdx = circularLaneShift + (blockStart & ~LANE_MASK);
    //     if (writeIdx < numBlocks) {
    //         passHist[writeIdx + digitOffset] =
    //             (laneId != LANE_MASK ? s_scan[tid] : 0) +
    //             (tid >= WARP_SIZE ? 
    //                 __shfl_sync(0xffffffff, s_scan[tid - 1], 0) : 0) +
    //             reduction;
    //     }

    //     reduction += s_scan[blockSize - 1];
    //     __syncthreads();
    // }
    uint32_t tidx = tid;
    for(; tidx<fullBlocksEnd; tidx += blockDim.x) {
        s_scan[tid] = passHist[tid + digitOffset];
        
        // Perform warp-level scan
        s_scan[tid] = InclusiveWarpScan(s_scan[tid]);
        __syncthreads();

        // Collect and scan warp totals
        if (tid < (blockDim.x >> LANE_LOG)) {
            s_scan[((tid + 1) << LANE_LOG) - 1] = ActiveInclusiveWarpScan(s_scan[((tid + 1) << LANE_LOG) - 1]);
        }
        __syncthreads();

        const uint32_t writeIdx = circularLaneShift + (tidx & ~LANE_MASK);

        passHist[writeIdx + digitOffset] =
            (getLaneId() != LANE_MASK ? s_scan[tid] : 0) +
            (tid >= WARP_SIZE ?
            s_scan[(tid & ~LANE_MASK) - 1] : 0) +
            reduction;

        reduction += s_scan[blockDim.x - 1];
        __syncthreads();
    }

    // Remaining elements handled similarly...
    uint32_t remainingElements = numBlocks - fullBlocksEnd;
    // reduction += s_scan[remainingElements - 1];

    if(tidx < numBlocks) {
        s_scan[tid] = passHist[tid + digitOffset];
    }

    s_scan[tid] = InclusiveWarpScan(s_scan[tid]);
    __syncthreads();

    if(tid < (blockDim.x >> LANE_LOG)) {
        s_scan[((tid + 1) << LANE_LOG) - 1] = ActiveInclusiveWarpScan(s_scan[((tid + 1) << LANE_LOG) - 1]);
    }
    __syncthreads();

    const uint32_t writeIdx = circularLaneShift + (tidx & ~LANE_MASK);
    if (writeIdx < numBlocks) {
        passHist[writeIdx + digitOffset] =
            (getLaneId() != LANE_MASK ? s_scan[tid] : 0) +
            (tid >= WARP_SIZE ?
            s_scan[(tid & ~LANE_MASK) - 1] : 0) +
            reduction;
    }

    // if (fullBlocksEnd + tid < numBlocks) {
    //     // Load remaining data with circular shift pattern
    //     s_scan[tid] = passHist[fullBlocksEnd + tid + digitOffset];    
    //     s_scan[tid] = InclusiveWarpScan(s_scan[tid]);
    //     __syncthreads();
   
    //     if (tid < blockDim.x / WARP_SIZE) {
    //         s_scan[((tid + 1) << LANE_LOG) - 1] = ActiveInclusiveWarpScan(s_scan[((tid + 1) << LANE_LOG) - 1]);
    //     }
    //     __syncthreads();

    //     // const uint32_t outputIdx = tid + (fullBlocksEnd & ~LANE_MASK);
    //     const uint32_t writeIdx = circularLaneShift + (fullBlocksEnd & ~LANE_MASK);
    //     if (writeIdx < numBlocks) {
    //         passHist[writeIdx + digitOffset] =
    //             (laneId != LANE_MASK ? s_scan[tid] : 0) +
    //             (tid >= WARP_SIZE ? 
    //                 s_scan[(tid & ~LANE_MASK) - 1] : 0) +
    //             reduction;
    //     }
    // }
    // __syncthreads();
    if(blockIdx.x == 1 && tid == numBlocks - 1) {
        printf("\nDigit[%u]\n", blockIdx.x);
        for(uint32_t i=0; i<blockDim.x; ++i) {
            printf("[%u %u] ", s_scan[i], i < numBlocks ? passHist[digitOffset + i] : 0);
        }
        printf("\n");
    }
}

template<typename T>
__global__ void RadixDownsweep(
    T* sort,              // Input array
    T* alt,               // Output array
    uint32_t* globalHist, // Global histogram
    uint32_t* passHist,   // Pass histogram
    uint32_t size,        // Total elements to sort
    uint32_t radixShift)  // Current radix shift amount
{
    // Shared memory layout
    // __shared__ uint32_t s_warpHistograms[N_THREADS * BIN_KEYS_PER_THREAD];  // blockDim.x * BIN_KEYS_PER_THREAD
    // __shared__ uint32_t s_localHistogram[RADIX];   // RADIX
    // volatile uint32_t* s_warpHist = &s_warpHistograms[WARP_INDEX << RADIX_LOG];

    // //clear shared memory
    // for (uint32_t i = threadIdx.x; i < BIN_HISTS_SIZE; i += blockDim.x)
    //     s_warpHistograms[i] = 0;

    // // Calculate thread's global position and valid key count
    // uint32_t thread_start = blockIdx.x * blockDim.x * BIN_KEYS_PER_THREAD + threadIdx.x;
    // uint32_t valid_keys = min(BIN_KEYS_PER_THREAD, 
    //                          (size - thread_start + WARP_SIZE - 1) / WARP_SIZE);
    
    // // Get ballot mask for partially filled warps
    // uint32_t ballot_mask = getBallotMask(thread_start, size);

    // // Clear shared memory
    // for (uint32_t i = threadIdx.x; i < BIN_HISTS_SIZE; i += blockDim.x) {
    //     s_warpHistograms[i] = 0;
    // }
    // __syncthreads();

    // // Load and convert keys
    // uint32_t keys[BIN_KEYS_PER_THREAD];
    // #pragma unroll
    // for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i) {
    //     uint32_t idx = thread_start + i * WARP_SIZE;
    //     keys[i] = (idx < size && i < valid_keys) ? toBits<T>(sort[idx]) : 0;
    // }
    // __syncthreads();

    // // WLMS - Warp Level Multi Split
    // uint32_t offsets[BIN_KEYS_PER_THREAD];
    // #pragma unroll
    // for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i) {
    //     unsigned warpFlags = ballot_mask;
        
    //     if (i < valid_keys) {
    //         #pragma unroll
    //         for (int k = 0; k < RADIX_LOG; ++k) {
    //             const bool t2 = keys[i] >> (k + radixShift) & 1;
    //             warpFlags &= (t2 ? 0 : ballot_mask) ^ __ballot_sync(ballot_mask, t2);
    //         }
    //     }

    //     const uint32_t bits = __popc(warpFlags & getLaneMaskLt());
    //     uint32_t preIncrementVal = 0;

    //     // Only the first thread in each digit group updates the histogram
    //     if (bits == 0 && i < valid_keys) {
    //         uint32_t digit = keys[i] >> radixShift & RADIX_MASK;
    //         preIncrementVal = atomicAdd(&s_warpHist[digit], __popc(warpFlags));
    //     }

    //     // Share the offset with other threads in the same digit group
    //     offsets[i] = __shfl_sync(ballot_mask, preIncrementVal, __ffs(warpFlags) - 1) + bits;
        
    //     // Validate offset is within bounds
    //     assert(offsets[i] < BIN_PART_SIZE);
    // }
    // __syncthreads();

    // // Exclusive prefix sum up the warp histograms
    // if (threadIdx.x < RADIX) {
    //     uint32_t reduction = s_warpHistograms[threadIdx.x];
    //     for (uint32_t i = threadIdx.x + RADIX; i < BIN_HISTS_SIZE; i += RADIX) {
    //         reduction += s_warpHistograms[i];
    //         s_warpHistograms[i] = reduction - s_warpHistograms[i];
    //     }

    //     s_warpHistograms[threadIdx.x] = InclusiveWarpScanCircularShift(reduction);
    // }
    // __syncthreads();

    // // Additional prefix sum processing
    // if (threadIdx.x < (RADIX >> WARP_LOG)) {
    //     s_warpHistograms[threadIdx.x << WARP_LOG] = 
    //         ActiveExclusiveWarpScan(s_warpHistograms[threadIdx.x << WARP_LOG]);
    // }
    // __syncthreads();

    // // Update offsets based on warp index
    // if (WARP_INDEX) {
    //     #pragma unroll 
    //     for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i) {
    //         if (i < valid_keys) {
    //             const uint32_t digit = keys[i] >> radixShift & RADIX_MASK;
    //             offsets[i] += s_warpHist[digit] + s_warpHistograms[digit];
    //         }
    //     }
    // }
    // else {
    //     #pragma unroll
    //     for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i) {
    //         if (i < valid_keys) {
    //             offsets[i] += s_warpHistograms[keys[i] >> radixShift & RADIX_MASK];
    //         }
    //     }
    // }

    // // Load block-level histogram data
    // if (threadIdx.x < RADIX) {
    //     s_localHistogram[threadIdx.x] = globalHist[threadIdx.x + (radixShift << 5)] +
    //         passHist[threadIdx.x * gridDim.x + blockIdx.x] - s_warpHistograms[threadIdx.x];
    // }
    // __syncthreads();

    // // Scatter keys to shared memory
    // #pragma unroll
    // for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i) {
    //     if (i < valid_keys) {
    //         s_warpHistograms[offsets[i]] = keys[i];
    //     }
    // }
    // __syncthreads();

    // // Final scatter to global memory
    // uint32_t block_items = min(blockDim.x * BIN_KEYS_PER_THREAD, 
    //                           size - blockIdx.x * blockDim.x * BIN_KEYS_PER_THREAD);
    
    // for (uint32_t i = threadIdx.x; i < block_items; i += blockDim.x) {
    //     uint32_t digit = s_warpHistograms[i] >> radixShift & RADIX_MASK;
    //     uint32_t global_idx = s_localHistogram[digit] + i;
    //     if (global_idx < size) {
    //         // alt[global_idx] = fromBits<T>(s_warpHistograms[i]);
    //     }
    // }
}