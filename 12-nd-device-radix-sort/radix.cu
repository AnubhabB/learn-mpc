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

// Vector type definitions
template<typename T>
struct alignas(16) Vector {  // Align to 16 bytes for optimal memory access
    T data[VectorTrait<T>::vector_size];
    
    __device__ __host__ T& operator[](int i) { return data[i]; }
    __device__ __host__ const T& operator[](int i) const { return data[i]; }
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
    const uint32_t numElemsInBlock // max number of elements being processed by this block
) {
    // Shared memory for histogram - two sections to avoid bank conflicts
    constexpr uint32_t sharedSize = RADIX * 2;
    __shared__ uint32_t s_globalHist[sharedSize];

    // Clear shared memory histogram
    #pragma unroll
    for (uint32_t i = threadIdx.x; i < sharedSize; i += blockDim.x)
        s_globalHist[i] = 0;
    __syncthreads();

    // Calculate this block's range
    const uint32_t block_start = blockIdx.x * numElemsInBlock;
    const uint32_t block_end = min(block_start + numElemsInBlock, size);
    const uint32_t elements_in_block = block_end - block_start;

    // Vector load based on types
    // constexpr uint32_t typesize = sizeof(T);
    using VecT = Vector<T>;
    constexpr uint32_t vec_size = VectorTrait<T>::vector_size;

    // Calculate number of full vectors - we are going to make an attempt to process 4 vectors at a time
    const uint32_t full_vecs = elements_in_block / vec_size;

    for (uint32_t i = threadIdx.x; i < full_vecs; i += blockDim.x) {
        const uint32_t idx = block_start / vec_size + i;
        const VecT vec_val = reinterpret_cast<const VecT*>(sort)[idx];
        
        #pragma unroll
        for (int j = 0; j < vec_size; j++) {
            uint32_t bits = toBits(vec_val[j]);
            atomicAdd(&s_globalHist[bits >> radixShift & RADIX_MASK], 1);
        }
    }

    // Process remaining elements
    const uint32_t vec_end = block_start + (full_vecs * vec_size);
    for (uint32_t i = threadIdx.x + vec_end; i < block_end; i += blockDim.x) {
        const T t = sort[i];
        uint32_t bits = toBits(t);
        atomicAdd(&s_globalHist[bits >> radixShift & RADIX_MASK], 1);
    }

    __syncthreads();

    // Reduce histograms and prepare for prefix sum
    for (uint32_t i = threadIdx.x; i < RADIX; i += blockDim.x) {
        s_globalHist[i] += s_globalHist[i + RADIX];
        passHist[i * gridDim.x + blockIdx.x] = s_globalHist[i];
        s_globalHist[i] = InclusiveWarpScanCircularShift(s_globalHist[i]);
    }   
    __syncthreads();

    // Perform warp-level scan
    if (threadIdx.x < (RADIX >> LANE_LOG))
        s_globalHist[threadIdx.x << LANE_LOG] = ActiveExclusiveWarpScan(s_globalHist[threadIdx.x << LANE_LOG]);
    __syncthreads();

    // Update global histogram with prefix sum results
    for (uint32_t i = threadIdx.x; i < RADIX; i += blockDim.x) {
        atomicAdd(
            &globalHist[i + (radixShift << LANE_LOG)], 
            s_globalHist[i] + 
            (getLaneId() ? 
                __shfl_sync(0xfffffffe, s_globalHist[i - 1], 1) : 0)
        );
    }
}


// TODO: optimize this with shared memory
__global__ void RadixScan(
    uint32_t* passHist,
    const uint32_t threadBlocks
) {
    __shared__ uint32_t s_scan[128];

    uint32_t reduction = 0;
    // Get ID of the next thread: 0 -> 1, 1 -> 2 ... 31 -> 0
    const uint32_t circularLaneShift = getLaneId() + 1 & LANE_MASK;
    // Each block is responsible for one digit - we are launching this with `RADIX` blocks
    const uint32_t digitOffset = blockIdx.x * threadBlocks;
    
    // Load this digit to shared memory
    // For `0`th digit we are getting `Block 0: 0`, `Block 1: 0` ...
    // if(threadIdx.x < threadBlocks) {
    //     if(threadIdx.x + digitOffset < 36) {
    //         printf("Bf: [%u %u %u] ", threadIdx.x + digitOffset, passHist[threadIdx.x + digitOffset], s_scan[threadIdx.x + digitOffset]);
    //     }
    //     s_scan[threadIdx.x] = passHist[threadIdx.x + digitOffset];
    //     if(threadIdx.x + digitOffset < 36) {
    //         printf("Af: [%u %u] ", threadIdx.x + digitOffset, s_scan[threadIdx.x]);
    //     }
    // }
    s_scan[threadIdx.x] = (threadIdx.x < threadBlocks) ? passHist[threadIdx.x + digitOffset] : 0;
    s_scan[threadIdx.x] = InclusiveWarpScan(s_scan[threadIdx.x]);
    __syncthreads();

    // if((blockIdx.x == 0) &&  threadIdx.x == 127) {
    //     printf("Block[%u]\n", blockIdx.x);
    //     for(uint32_t i=0; i<128; i++) {
    //         printf("%u ", s_scan[i]);
    //     }
    // }
    // Now, for every warp of the block, update the last element in the block with a `inclusive prefix sum`
    // Only one thread per warp is acting on this
    if(threadIdx.x < blockDim.x >> LANE_LOG) {
        uint32_t warp_last_idx = (threadIdx.x + 1 << LANE_LOG) - 1;
        s_scan[warp_last_idx]  = ActiveInclusiveWarpScan(s_scan[warp_last_idx]); 
    }
    __syncthreads();

    
    // // Current thread position within this warp
    // const uint32_t lane_id = getLaneId();
    // // In the `Upsweep` kernel we have computed `passHist` per threadBlock
    // // So, the current digit to process is @ blockIdx * threadBlocks
    // const uint32_t digit_offset = blockIdx.x * threadBlocks;
    
    // // Process in chunks of WARP_SIZE
    // const uint32_t num_warps = (threadBlocks + WARP_SIZE - 1) / WARP_SIZE;
    // uint32_t running_sum = 0;
    
    // // Process each warp-sized chunk
    // for (uint32_t warp = 0; warp < num_warps; warp++) {
    //     const uint32_t start_idx = warp * WARP_SIZE;
    //     const uint32_t local_idx = start_idx + lane_id;
        
    //     // Load and scan within warp
    //     uint32_t val = 0;
    //     if (local_idx < threadBlocks) {
    //         val = passHist[digit_offset + local_idx];
    //     }
        
    //     // Perform inclusive scan within warp
    //     val = InclusiveWarpScan(val);
        
    //     // Add running sum from previous iterations
    //     val += running_sum;
        
    //     // Store result if within bounds
    //     if (local_idx < threadBlocks) {
    //         passHist[digit_offset + local_idx] = val;
    //     }
        
    //     // Update running sum for next iteration
    //     // Get the last valid value in this warp
    //     uint32_t warp_last = __shfl_sync(0xffffffff, val, min(threadBlocks - start_idx, WARP_SIZE) - 1);
    //     if (lane_id == 0) {
    //         running_sum = warp_last;
    //     }
    // }
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