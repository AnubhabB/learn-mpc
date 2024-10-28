#include <stdio.h>
#include <stdint.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define RADIX               256             //Number of digit bins
#define WARP_SIZE           32
#define LANE_LOG            5               // LANE_LOG = 5 since 2^5 = 32 = warp size
#define LANE_MASK           (WARP_SIZE - 1)
#define RADIX_MASK          (RADIX - 1)     //Mask of digit bins, to extract digits

__device__ __forceinline__ uint32_t getLaneId() {
    uint32_t laneId;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
    return laneId;
}

// Prefix sum
// Performs an inclusive scan operation within a single warp
// Uses butterfly/sequential addressing pattern for efficiency
__device__ __forceinline__ uint32_t InclusiveWarpScan(uint32_t val) {
    #pragma unroll
    for (int offset = 1; offset <= WARP_SIZE; offset <<= 1) {
        const uint32_t t = __shfl_up_sync(0xffffffff, val, offset);
        if (getLaneId() >= offset) val += t;
    }

    return val;
}

// Circular shift prefix sum
__device__ __forceinline__ uint32_t InclusiveWarpScanCircularShift(uint32_t val) {
    #pragma unroll
    for (int offset = 1; offset <= WARP_SIZE; offset <<= 1) {
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
    for (int offset = 1; offset <= WARP_SIZE; offset <<= 1) {
        if (offset >= active_threads) break;  // Early termination
        const uint32_t t = __shfl_up_sync(mask, val, offset);
        if (getLaneId() >= offset) val += t;
    }

    return val;
}

// Prefix sum of active threads ot including itself
__device__ __forceinline__ uint32_t ActiveExclusiveWarpScan(uint32_t val) {
    const uint32_t mask = __activemask();
    #pragma unroll
    for (int offset = 1; offset <= WARP_SIZE; offset <<= 1) {
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


template<typename T>
__global__ void RadixUpsweep(
    T* sort,
    uint32_t* globalHist,
    uint32_t* passHist,
    const uint32_t size,
    const uint32_t radixShift,
    const uint32_t partSize, // max number of elements being processed by this block
    const uint32_t vecPartSize // max number of `vector elements` in block
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
    const uint32_t block_start = blockIdx.x * partSize;
    const uint32_t block_end = min(block_start + partSize, size);
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
    const uint32_t lane_id = getLaneId();
    const uint32_t digit_offset = blockIdx.x * threadBlocks;
    
    // Process in chunks of WARP_SIZE
    const uint32_t num_warps = (threadBlocks + WARP_SIZE - 1) / WARP_SIZE;
    uint32_t running_sum = 0;
    
    // Process each warp-sized chunk
    for (uint32_t warp = 0; warp < num_warps; warp++) {
        const uint32_t start_idx = warp * WARP_SIZE;
        const uint32_t local_idx = start_idx + lane_id;
        
        // Load and scan within warp
        uint32_t val = 0;
        if (local_idx < threadBlocks) {
            val = passHist[digit_offset + local_idx];
        }
        
        // Perform inclusive scan within warp
        val = InclusiveWarpScan(val);
        
        // Add running sum from previous iterations
        val += running_sum;
        
        // Store result if within bounds
        if (local_idx < threadBlocks) {
            passHist[digit_offset + local_idx] = val;
        }
        
        // Update running sum for next iteration
        // Get the last valid value in this warp
        uint32_t warp_last = __shfl_sync(0xffffffff, val, min(threadBlocks - start_idx, WARP_SIZE) - 1);
        if (lane_id == 0) {
            running_sum = warp_last;
        }
    }
}

// __device__ __forceinline__ uint32_t warpInclusiveScan(uint32_t val) {
//     #pragma unroll
//     for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
//         uint32_t temp = __shfl_up_sync(0xffffffff, val, offset);
//         if (getLaneId() >= offset) {
//             val += temp;
//         }
//     }
//     return val;
// }

// __global__ void RadixScan(
//     uint32_t* passHist,
//     const uint32_t threadBlocks
// ) {
//     extern __shared__ uint32_t s_scan[];
    
//     const uint32_t lane_id = getLaneId();
//     const uint32_t digit_offset = blockIdx.x * threadBlocks;

//     // Special case for small thread blocks
//     if (threadBlocks <= WARP_SIZE) {
//         if (lane_id < threadBlocks) {
//             uint32_t val = passHist[digit_offset + lane_id];
            
//             // Perform scan within the first warp
//             val = warpInclusiveScan(val);
            
//             // Write back
//             passHist[digit_offset + lane_id] = val;
//         }
//         return;
//     }
    
//     // Regular case for larger thread blocks
//     const uint32_t warp_id = threadIdx.x >> LANE_LOG;
    
//     // Step 1: Load data into shared memory
//     uint32_t val = 0;
//     if (threadIdx.x < threadBlocks) {
//         val = passHist[digit_offset + threadIdx.x];
//     }
//     s_scan[threadIdx.x] = val;
//     __syncthreads();
    
//     // Step 2: Perform intra-warp scan
//     val = warpInclusiveScan(val);
//     s_scan[threadIdx.x] = val;
//     __syncthreads();
    
//     // Step 3: Collect last elements from each warp
//     if (lane_id == WARP_SIZE-1 && warp_id < (threadBlocks + WARP_SIZE - 1) / WARP_SIZE) {
//         uint32_t warp_result = s_scan[threadIdx.x];
//         s_scan[blockDim.x + warp_id] = warp_result;
//     }
//     __syncthreads();
    
//     // Step 4: Scan warp results using first warp
//     if (warp_id == 0 && lane_id < (threadBlocks + WARP_SIZE - 1) / WARP_SIZE) {
//         uint32_t warp_sum = s_scan[blockDim.x + lane_id];
//         warp_sum = warpInclusiveScan(warp_sum);
//         s_scan[blockDim.x + lane_id] = warp_sum;
//     }
//     __syncthreads();
    
//     // Step 5: Add warp scan results back
//     if (threadIdx.x < threadBlocks) {
//         if (warp_id > 0) {
//             val += s_scan[blockDim.x + warp_id - 1];
//         }
//         passHist[digit_offset + threadIdx.x] = val;
//     }
// }