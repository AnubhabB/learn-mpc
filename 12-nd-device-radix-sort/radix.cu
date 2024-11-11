#include <stdio.h>
#include <stdint.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define N_THREADS           256
#define RADIX               256             //Number of digit bins
#define WARP_SIZE           32
#define LANE_LOG            5               // LANE_LOG = 5 since 2^5 = 32 = warp size
#define RADIX_LOG           8               // 2^8 = 256

#define LANE_MASK           (WARP_SIZE - 1)
#define RADIX_MASK          (RADIX - 1)     //Mask of digit bins, to extract digits
#define WARP_INDEX          (threadIdx.x >> LANE_LOG)

#define BIN_KEYS_PER_THREAD 15

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
        if (isinf(val)) {
            // Handle infinity
            return val < 0.0f ? 0 : 0xFFFFFFFF;
        }
        uint32_t bits = __float_as_uint(val);
        
        return (bits & 0x80000000) ? ~bits : bits ^ 0x80000000;
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
    } else if constexpr (std::is_same<T, __half>::value) {
        return __float2half(INFINITY);
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        return __float2bfloat16(INFINITY);
    } else if constexpr (std::is_same<T, int64_t>::value) {
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
    T* keys,
    uint32_t* globalHist,
    uint32_t* passHist,
    const uint32_t size,
    const uint32_t radixShift,
    const uint32_t maxElemInBlock // max number of elements being processed by this block
) {
    // uint32_t printBlock = 0;
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

    // if (blockIdx.x == printBlock && (threadIdx.x == 0 || threadIdx.x == 15)) {
    //     printf("Thread[%u]: maxElemInBlock[%u] block_start[%u] block_end[%u] elements_in_block[%u]\n", threadIdx.x, maxElemInBlock, block_start, block_end, elements_in_block);
    // }

    // if (threadIdx.x == 0 && blockIdx.x == printBlock) {
    //     // printf("\nBlock[%u]: ", blockIdx.x);
    //     for (uint32_t i=block_start; i<block_end; ++i) {
    //         printf("%u ", i]);
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
        
        if (idx < vec_end) {
            // if (blockIdx.x == printBlock) {
            //     printf("Idx[%u] ", idx);
            // }
            #pragma unroll
            for (int j = 0; j < vec_size; ++j) {
                uint32_t bits = toBits(keys[idx + j]);
                // if (blockIdx.x == printBlock) {
                //     printf("sort[%u %u]: [%u %u %u] ", idx, idx + j, sort[idx + j], bits, bits >> radixShift & RADIX_MASK);
                // }
                atomicAdd(&s_globalHist[bits >> radixShift & RADIX_MASK], 1);
            }
        }
    }
    
    // Process remaining elements
    for (uint32_t i = threadIdx.x + vec_end; i < block_end; i += blockDim.x) {
        uint32_t bits = toBits(keys[i]);
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
    // if (threadIdx.x == 15 && blockIdx.x == printBlock) {
    //     printf("\nBlockHist[%u]: ", blockIdx.x);
    //     for (uint32_t i=0; i<RADIX; ++i) {
    //         printf("[%u %u %u]\n", i, passHist[RADIX * blockIdx.x + i], s_globalHist[i]);
    //     }
    //     printf("--\n");
    // }

    // Perform warp-level scan - for first thread in each warp
    if (threadIdx.x < (RADIX >> LANE_LOG))
        s_globalHist[threadIdx.x << LANE_LOG] = ActiveExclusiveWarpScan(s_globalHist[threadIdx.x << LANE_LOG]);
    __syncthreads();

    // if (threadIdx.x == 15 && blockIdx.x == printBlock) {
    //     printf("\nBlockHist[%u]: ", blockIdx.x);
    //     for (uint32_t i=0; i<RADIX; ++i) {
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

    // if (threadIdx.x == 15 && blockIdx.x == printBlock) {
    //     printf("\nBlockHist[%u]: ", blockIdx.x);
    //     for (uint32_t i=0; i<RADIX; ++i) {
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

    extern __shared__ uint32_t s_scan[];

    // Circular shift within warp - this helps reduce bank conflicts
    // Get ID of the next thread: getLaneId(): 0 -> 1, 1 -> 2 ... 31 -> 0
    const uint32_t circularLaneShift = (laneId + 1) & LANE_MASK;

    // Where does the digit start
    const uint32_t digitOffset = blockIdx.x * numBlocks;

    // Calculate the number of full block-sized chunks we need to process
    const uint32_t fullBlocksEnd = (numBlocks / blockSize) * blockSize;
    
    // Running sum for carrying over between iterations
    uint32_t reduction = 0;

    uint32_t tidx = tid;
    for (; tidx<fullBlocksEnd; tidx += blockDim.x) {
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
    if (tidx < numBlocks) {
        s_scan[tid] = passHist[tid + digitOffset];
    }

    s_scan[tid] = InclusiveWarpScan(s_scan[tid]);
    __syncthreads();

    if (tid < (blockDim.x >> LANE_LOG)) {
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

    // if (blockIdx.x == 30 && tid == numBlocks - 1) {
    //     printf("\nDigit[%u]\n", blockIdx.x);
    //     for (uint32_t i=0; i<blockDim.x; ++i) {
    //         printf("[%u %u] ", s_scan[i], i < numBlocks ? passHist[digitOffset + i] : 0);
    //     }
    //     printf("\n");
    // }
}

template<typename T>
__global__ void RadixDownsweep(
    T* keys,                        // Input array
    T* keysAlt,                     // Output array
    uint32_t* vals,                 // Optional values to be sorted
    uint32_t* valsAlt,               // Output array
    const uint32_t* globalHist,     // Global histogram
    const uint32_t* passHist,       // Pass histogram
    const uint32_t size,            // Total elements to sort
    const uint32_t radixShift,      // current radixShift bit
    const uint32_t maxElemInBlock,  // Number of elements processed per partition/ block
    const uint32_t histSize,        // size of the histogram initialized externally
    const uint32_t numKeysPerThread,// real number of keys processed by each thread. Max would be `BIN_KEYS_PER_THREAD`
    const bool sortIndex            // if set to true, attempt to sort the indices
) {
    // Shared memory layout
    // `s_tmp` would be used for `s_warpHistograms` and later for `s_keys` and `s_values` (optional)  
    extern __shared__ uint32_t s_tmp[]; // Number of warps needed = ceil(N / (WARP_SIZE * numKeysPerThread)); Size: num_warps * RADIX
    __shared__ uint32_t s_localHistogram[RADIX];
    volatile uint32_t* s_warpHist = &s_tmp[WARP_INDEX << RADIX_LOG];

    // for warp histogram temp storage
    uint32_t* s_warpHistograms = s_tmp;

    // The partition offset of keys to work with
    const uint32_t blockOffset = blockIdx.x * maxElemInBlock;
    const uint32_t tidInWarp   = ((numKeysPerThread << LANE_LOG) * WARP_INDEX) + getLaneId();

    //clear shared memory
    for (uint32_t i = threadIdx.x; i < histSize; i += blockDim.x)
        s_tmp[i] = 0;

    uint32_t threadStore[BIN_KEYS_PER_THREAD]; // local store for max keys per thread to be later used for indices
    // uint32_t threadVals[BIN_KEYS_PER_THREAD]; // local store for max values per thread. This will remain un-initialized when `!sortIdx`
    uint16_t offsets[BIN_KEYS_PER_THREAD];

    T* threadKeys = reinterpret_cast<T*>(threadStore);
    //load keys
    // We are going to be processing `BIN_KEYS_PER_THREAD` keys per thread
    // The starting location of the each key =
    //        Block in which a key belongs (block index * maxElemenInBlock) +
    //        (In a block, offset of a key with respect to warps
    //              Number of elements per warp (BIN_KEYS_PER_THREAD * WARP_SIZE) * Warp Index) +
    //         LaneId
    //
    // To handle input sizes not perfect multiples of the partition tile size,
    // load "dummy" keys, which are keys with the highest possible digit.
    // Because of the stability of the sort, these keys are guaranteed to be 
    // last when scattered. This allows for effortless divergence free sorting
    // of the final partition.
    #pragma unroll
    for (uint32_t i=0, t=blockOffset + tidInWarp; i<numKeysPerThread;++i, t+=WARP_SIZE) {
        threadKeys[i] = t < size ? keys[t] : getTypeMax<T>();
    }
    __syncthreads();

    // WLMS (warp-level multi-split) Ashkiani et al (https://arxiv.org/pdf/1701.01189)
    // Computes warp level histogram for digits
    #pragma unroll
    for (uint32_t i = 0; i < numKeysPerThread; ++i) {
        uint32_t bitval = toBits<T>(threadKeys[i]);

        // creating mask for threads in a warp that have same bit value as keys[i]
        unsigned warpFlags = 0xffffffff;
        #pragma unroll
        for (int k = 0; k < RADIX_LOG; ++k) {
            // true if `radixShift + kth` position is 1
            const bool t2 = (bitval >> (k + radixShift)) & 1;
            warpFlags &= (t2 ? 0 : 0xffffffff) ^ __ballot_sync(0xffffffff, t2);
        }

        // Counts the number of bits set to `1` in the current warp
        const uint32_t bits = __popc(warpFlags & getLaneMaskLt());
        uint32_t preIncrementVal;
        // Update histogram count only once per warp
        if (bits == 0) {
            preIncrementVal = atomicAdd((uint32_t*)&s_warpHist[(bitval >> radixShift) & RADIX_MASK], __popc(warpFlags));
        }

        offsets[i] = __shfl_sync(0xffffffff, preIncrementVal, __ffs(warpFlags) - 1) + bits;
    }
    __syncthreads();

    // if (threadIdx.x == 63 && blockIdx.x == 0) {
    //     printf("\n=== Verification for Thread %u with radixShift[%u] ===\n", threadIdx.x, radixShift);
    //     for (int i = 0; i < BIN_KEYS_PER_THREAD; i++) {
    //         printf("Key[%d]: %u -> Offset: %u | (keys[i] >> radixShift) & RADIX_MASK: %u\n", i, keys[i], offsets[i], (keys[i] >> radixShift) & RADIX_MASK);
            
    //         // Verify offset is within bounds
    //         if (offsets[i] >= size) {
    //             printf("WARNING: Offset %u exceeds size %u\n", offsets[i], size);
    //         }
            
    //         // Verify offset ordering within same radix digit
    //         if (i > 0 && ((keys[i] >> radixShift & RADIX_MASK) == 
    //                     (keys[i-1] >> radixShift & RADIX_MASK))) {
    //             if (offsets[i] <= offsets[i-1]) {
    //                 printf("WARNING: Non-monotonic offsets for same digit: %u <= %u\n",
    //                     offsets[i], offsets[i-1]);
    //             }
    //         }
    //     }
    //     // printf("==== Debug for Thread %u, Key[%d] ====\n", keyIndex);
    //     // printf("Key Value: %u\n", keys[keyIndex]);
    //     // printf("Radix Digit: %u\n", keys[keyIndex] >> radixShift & RADIX_MASK);
    //     // printf("Warp Flags: 0x%08x\n", warpFlags);
    //     // printf("Bits before current thread: %u\n", bits);
    //     // printf("Pre-increment value: %u\n", preIncrementVal);
    //     // printf("Final offset: %u\n", offsets[keyIndex]);
    //     // printf("================================\n");
    // }
    // After initial WLMS (what we saw before)
    // if (threadIdx.x == 31 && blockIdx.x == 0) {
    //     printf("\n=== After WLMS ===\n");
    //     printf("s_warpHistograms[0]: %u\n", s_warpHistograms[0]);
    //     printf("s_warpHistograms[1]: %u\n", s_warpHistograms[1]);
    //     printf("s_warpHistograms[2]: %u\n", s_warpHistograms[2]);
    //     printf("s_warpHistograms[3]: %u\n", s_warpHistograms[3]);
    //     printf("s_warpHistograms[4]: %u\n", s_warpHistograms[4]);
    //     printf(" ... \n");
    //     printf("s_warpHistograms[33]: %u\n", s_warpHistograms[33]);
    //     printf("s_warpHistograms[34]: %u\n", s_warpHistograms[34]);
    //     printf("s_warpHistograms[35]: %u\n", s_warpHistograms[35]);
    //     printf(" ... \n");
    //     printf("s_warpHistograms[41]: %u\n", s_warpHistograms[41]);
    //     printf("s_warpHistograms[42]: %u\n", s_warpHistograms[42]);
    // }
    // __syncthreads();
    // At this stage we have `warp-level` histograms
    // exclusive prefix sum up the warp histograms
    if (threadIdx.x < RADIX) {
        uint32_t reduction = s_warpHistograms[threadIdx.x];
        for (uint32_t i = threadIdx.x + RADIX; i < histSize; i += RADIX) {
            reduction += s_warpHistograms[i];
            s_warpHistograms[i] = reduction - s_warpHistograms[i];
        }

        //begin the exclusive prefix sum across the reductions
        s_warpHistograms[threadIdx.x] = InclusiveWarpScanCircularShift(reduction);
    }
    __syncthreads();
    // uint32_t digitsPerThread = RADIX / blockDim.x  + (RADIX % blockDim.x != 0);
    // uint32_t startIdx = threadIdx.x * digitsPerThread;
    // uint32_t endIdx = min(startIdx + digitsPerThread, RADIX);

    // if (startIdx < RADIX) {
    //     for (uint32_t idx = startIdx; idx < endIdx; ++idx) {
    //         uint32_t reduction = s_warpHistograms[idx];
    //         // Process column for this digit
    //         for (uint32_t i = idx + RADIX; i < histSize; i += RADIX) {
    //             reduction += s_warpHistograms[i];
    //             s_warpHistograms[i] = reduction - s_warpHistograms[i];
    //         }
            
    //         // All threads in warp participate in scan
    //         s_warpHistograms[idx] = InclusiveWarpScanCircularShift(reduction);
    //     }
    // }
    // __syncthreads();

    // if (threadIdx.x == 31 && blockIdx.x == 0) {
    //     printf("\n=== Before warp boundary update ===\n");
    //     printf("s_warpHistograms[0]: %u\n", s_warpHistograms[0]);
    //     printf("s_warpHistograms[1]: %u\n", s_warpHistograms[1]);
    //     printf("s_warpHistograms[2]: %u\n", s_warpHistograms[2]);
    //     printf("s_warpHistograms[3]: %u\n", s_warpHistograms[3]);
    //     printf("s_warpHistograms[4]: %u\n", s_warpHistograms[4]);
    //     printf(" ... \n");
    //     printf("s_warpHistograms[33]: %u\n", s_warpHistograms[33]);
    //     printf("s_warpHistograms[34]: %u\n", s_warpHistograms[34]);
    //     printf("s_warpHistograms[35]: %u\n", s_warpHistograms[35]);
    //     printf(" ... \n");
    //     printf("s_warpHistograms[41]: %u\n", s_warpHistograms[41]);
    //     printf("s_warpHistograms[42]: %u\n", s_warpHistograms[42]);
    // }
    // __syncthreads();

    // if (threadIdx.x == 63 && blockIdx.x == 0) {
    //     printf("\n=== Verification for Thread %u with radixShift[%u] After Exclusive prefix sum ===\n", threadIdx.x, radixShift);
    //     for (uint32_t i = 0; i < RADIX; ++i) {
    //         printf("[%u %u] ", i, s_warpHistograms[i]);
    //     }
    //     printf("\n\n");
    // }
    
    // Update the first threads of warps
    if (threadIdx.x < (RADIX >> LANE_LOG)) {
        uint32_t val = s_warpHistograms[threadIdx.x << LANE_LOG];
        uint32_t result = ActiveExclusiveWarpScan(val);
        //  if (blockIdx.x == 0) {
        //     printf("Thread %d: Index %d, Input %u, Output %u\n", 
        //         threadIdx.x, threadIdx.x << LANE_LOG, val, result);
        // }
        s_warpHistograms[threadIdx.x << LANE_LOG] = result;

    }
    __syncthreads();
    // After first phase (warp boundary update)
    // if (threadIdx.x == 31 && blockIdx.x == 0) {
    //     printf("\n=== After Warp Boundary Update ===\n");
    //     printf("s_warpHistograms[0]: %u\n", s_warpHistograms[0]);
    //     printf("s_warpHistograms[1]: %u\n", s_warpHistograms[1]);
    //     printf("s_warpHistograms[2]: %u\n", s_warpHistograms[2]);
    //     printf("s_warpHistograms[3]: %u\n", s_warpHistograms[3]);
    //     printf("s_warpHistograms[4]: %u\n", s_warpHistograms[4]);
    //     printf(" ... \n");
    //     printf("s_warpHistograms[33]: %u\n", s_warpHistograms[33]);
    //     printf("s_warpHistograms[34]: %u\n", s_warpHistograms[34]);
    //     printf("s_warpHistograms[35]: %u\n", s_warpHistograms[35]);
    //     printf(" ... \n");
    //     printf("s_warpHistograms[41]: %u\n", s_warpHistograms[41]);
    //     printf("s_warpHistograms[42]: %u\n", s_warpHistograms[42]);
    // }
    // __syncthreads();

    if (threadIdx.x < RADIX && getLaneId())
        s_warpHistograms[threadIdx.x] += __shfl_sync(0xfffffffe, s_warpHistograms[threadIdx.x - 1], 1);
    __syncthreads();

    // After propagation phase
    // if (threadIdx.x == 31 && blockIdx.x == 0) {
    //     printf("\n=== After Propagation ===\n");
    //     printf("s_warpHistograms[0]: %u\n", s_warpHistograms[0]);
    //     printf("s_warpHistograms[1]: %u\n", s_warpHistograms[1]);
    //     printf("s_warpHistograms[2]: %u\n", s_warpHistograms[2]);
    //     printf("s_warpHistograms[3]: %u\n", s_warpHistograms[3]);
    //     printf("s_warpHistograms[4]: %u\n", s_warpHistograms[4]);
    //     printf(" ... \n");
    //     printf("s_warpHistograms[33]: %u\n", s_warpHistograms[33]);
    //     printf("s_warpHistograms[34]: %u\n", s_warpHistograms[34]);
    //     printf("s_warpHistograms[35]: %u\n", s_warpHistograms[35]);
    //     printf(" ... \n");
    //     printf("s_warpHistograms[41]: %u\n", s_warpHistograms[41]);
    //     printf("s_warpHistograms[42]: %u\n", s_warpHistograms[42]);
    // }
    // __syncthreads();
    // if (threadIdx.x == 63 && blockIdx.x == 0) {
    //     printf("\n=== Verification for Thread %u with radixShift[%u] After Exclusive prefix sum ===\n", threadIdx.x, radixShift);
    //     for (uint32_t i = 0; i < RADIX; ++i) {
    //         printf("[%u %u] ", i, s_warpHistograms[i]);
    //     }
    //     printf("\n\n");
    // }

    //update offsets
    if (WARP_INDEX) {
        #pragma unroll
        for (uint32_t i = 0; i < numKeysPerThread; ++i) {
            const uint32_t t2 = toBits(threadKeys[i]) >> radixShift & RADIX_MASK;
            offsets[i] += s_warpHist[t2] + s_warpHistograms[t2];
        }
    } else {
        #pragma unroll
        for (uint32_t i = 0; i < numKeysPerThread; ++i)
            offsets[i] += s_warpHistograms[toBits(threadKeys[i]) >> radixShift & RADIX_MASK];
    }

    //load in threadblock reductions
    #pragma unroll
    for (uint32_t i=threadIdx.x; i<RADIX; i+=blockDim.x) {
        s_localHistogram[i] = globalHist[i + (radixShift << LANE_LOG)] +
            passHist[i * gridDim.x + blockIdx.x] - s_warpHistograms[i];
    }
    __syncthreads();

    // if (blockIdx.x == 0 && threadIdx.x == blockDim.x - 1) {
    //     printf("Block level histogram: ============================\n");
    //     for (uint32_t i=0; i<RADIX/2; ++i) {
    //         // printf("[%u %u %u %u %u] ", i, s_localHistogram[i], globalHist[i], passHist[i * gridDim.x], passHist[i * gridDim.x + 1]);
    //         printf("Digit %u globalHist: %u passHists: [%u %u] s_warpHistogram: %u s_localHistogram: %u\n", i, globalHist[i], passHist[i * gridDim.x], passHist[i * gridDim.x + 1], s_warpHistograms[i], s_localHistogram[i]);
    //     }
    //     printf("\n=================================================\n");
    // }

    // `s_tmp` has done with it's job as warp histogram bookkeeper
    // let's re-use it for our keys
    T* s_keys = reinterpret_cast<T*>(s_tmp);

    // scatter keys into shared memory
    #pragma unroll
    for (uint32_t i = 0; i < numKeysPerThread; ++i) {
        s_keys[offsets[i]] = threadKeys[i];
    }
    __syncthreads();

    //scatter runs of keys into device memory
    uint8_t digits[BIN_KEYS_PER_THREAD];
    // if (blockIdx.x < gridDim.x - 1) {
    uint32_t partSize = size - blockOffset;
    #pragma unroll
    for(uint32_t i=0, t=threadIdx.x; i<BIN_KEYS_PER_THREAD; ++i, t += blockDim.x) {
        if (i < numKeysPerThread && t < partSize) {
            digits[i] = toBits<T>(s_keys[t]) >> radixShift & RADIX_MASK;
            keysAlt[s_localHistogram[digits[i]] + t] = s_keys[t];
        }
    }

    // Now, we are done with the sorting of `keys`
    // If that's all we need, return
    if (!sortIndex)
        return;

    __syncthreads(); // this is required only if we proceed with sorting of indices
    // `s_tmp` has done with it's job as warp histogram bookkeeper & keys
    // let's re-use it for our vals
    uint32_t* s_vals = reinterpret_cast<uint32_t*>(s_tmp);
    uint32_t* threadVals = reinterpret_cast<uint32_t*>(threadStore);

    // Load indices into registers
    #pragma unroll
    for (uint32_t i = 0, t = blockOffset + tidInWarp; i < numKeysPerThread; ++i, t += WARP_SIZE) {
        threadVals[i] = t < size ? vals[t] : size; // `size` is a placeholder here > max index
    }
    __syncthreads();

    // scatter keys into shared memory
    #pragma unroll
    for (uint32_t i = 0; i < numKeysPerThread; ++i) {
        s_vals[offsets[i]] = threadVals[i];
    }
    __syncthreads();

    #pragma unroll
    for(uint32_t i=0, t=threadIdx.x; i<BIN_KEYS_PER_THREAD; ++i, t += blockDim.x) {
        if (i < numKeysPerThread && t < partSize) {
            valsAlt[s_localHistogram[digits[i]] + t] = s_vals[t];
        }
    }
    // __syncthreads();
    // #pragma unroll
    // for (uint32_t i = 0, t = getLaneId() + BIN_SUB_PART_START + BIN_PART_START;
    //     i < BIN_KEYS_PER_THREAD;
    //     ++i, t += LANE_COUNT)
    // {
    //     keys[i] = sortPayload[t];
    // }
    // } else {
    //     const uint32_t finalPartSize = size - blockOffset;
    //     // Last dim might have lesser number of elements
    //     for (uint32_t i=0, t=threadIdx.x; i<BIN_KEYS_PER_THREAD; ++i, t += blockDim.x) {
    //         if (t < finalPartSize) {
    //             digits[i] = toBits<T>(s_keys[t]) >> radixShift & RADIX_MASK;
    //             keysAlt[s_localHistogram[digits[i]] + t] = s_keys[t];
    //         }
    //     }

    //     __syncthreads();
    // }

    //scatter runs of keys into device memory
    // uint8_t digits[BIN_KEYS_PER_THREAD];
    // if (blockIdx.x < gridDim.x - 1)
    // {
    //     //store the digit of key in register
    //     #pragma unroll
    //     for (uint32_t i = 0, t = threadIdx.x; i < BIN_KEYS_PER_THREAD;
    //         ++i, t += blockDim.x)
    //     {
    //         digits[i] = s_warpHistograms[t] >> radixShift & RADIX_MASK;
    //         alt[s_localHistogram[digits[i]] + t] = s_warpHistograms[t];
    //     }
    //     __syncthreads();

    //     //Load payloads into registers
    //     #pragma unroll
    //     for (uint32_t i = 0, t = getLaneId() + BIN_SUB_PART_START + BIN_PART_START;
    //         i < BIN_KEYS_PER_THREAD;
    //         ++i, t += LANE_COUNT)
    //     {
    //         keys[i] = sortPayload[t];
    //     }

    //     //scatter payloads into shared memory
    //     #pragma unroll
    //     for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
    //         s_warpHistograms[offsets[i]] = keys[i];
    //     __syncthreads();

    //     //Scatter the payloads into device
    //     #pragma unroll
    //     for (uint32_t i = 0, t = threadIdx.x; i < BIN_KEYS_PER_THREAD;
    //         ++i, t += blockDim.x)
    //     {
    //         altPayload[s_localHistogram[digits[i]] + t] = s_warpHistograms[t];
    //     }
    // }

    // if (blockIdx.x == gridDim.x - 1)
    // {
    //     const uint32_t finalPartSize = size - BIN_PART_START;
    //     //store the digit of key in register
    //     #pragma unroll
    //     for (uint32_t i = 0, t = threadIdx.x; i < BIN_KEYS_PER_THREAD;
    //         ++i, t += blockDim.x)
    //     {
    //         if (t < finalPartSize)
    //         {
    //             digits[i] = s_warpHistograms[t] >> radixShift & RADIX_MASK;
    //             alt[s_localHistogram[digits[i]] + t] = s_warpHistograms[t];
    //         }
    //     }
    //     __syncthreads();

    //     //Load payloads into registers
    //     #pragma unroll
    //     for (uint32_t i = 0, t = getLaneId() + BIN_SUB_PART_START + BIN_PART_START;
    //         i < BIN_KEYS_PER_THREAD;
    //         ++i, t += LANE_COUNT)
    //     {
    //         if(t < size)
    //             keys[i] = sortPayload[t];
    //     }

    //     //scatter payloads into shared memory
    //     #pragma unroll
    //     for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
    //         s_warpHistograms[offsets[i]] = keys[i];
    //     __syncthreads();

    //     //Scatter the payloads into device
    //     #pragma unroll
    //     for (uint32_t i = 0, t = threadIdx.x; i < BIN_KEYS_PER_THREAD;
    //         ++i, t += blockDim.x)
    //     {
    //         if(t < finalPartSize)
    //             altPayload[s_localHistogram[digits[i]] + t] = s_warpHistograms[t];
    //     }
    // }
}