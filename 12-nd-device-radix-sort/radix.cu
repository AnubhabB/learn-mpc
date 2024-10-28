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
    for (int i = 1; i <= 16; i <<= 1) {
        const uint32_t t = __shfl_up_sync(0xffffffff, val, i);
        if (getLaneId() >= i) val += t;
    }

    return val;
}

// Circular shift prefix sum
__device__ __forceinline__ uint32_t InclusiveWarpScanCircularShift(uint32_t val) {
    // 16 = LANE_COUNT >> 1
    #pragma unroll
    for (int i = 1; i <= 16; i <<= 1) {
        const uint32_t t = __shfl_up_sync(0xffffffff, val, i, 32);
        if (getLaneId() >= i) val += t;
    }

    return __shfl_sync(0xffffffff, val, getLaneId() + LANE_MASK & LANE_MASK);
}

// Helper for active warp scan (used for inter-warp scan)
__device__ __forceinline__ uint32_t ActiveInclusiveWarpScan(uint32_t val) {
    const uint32_t mask = __activemask();
    #pragma unroll
    for (int i = 1; i <= 16; i <<= 1)
    {
        const uint32_t t = __shfl_up_sync(mask, val, i, 32);
        if (getLaneId() >= i) val += t;
    }

    return val;
}

// Prefix sum of active threads ot including itself
__device__ __forceinline__ uint32_t ActiveExclusiveWarpScan(uint32_t val) {
    const uint32_t mask = __activemask();
    // 16 = LANE_COUNT >> 1
    #pragma unroll
    for (int i = 1; i <= 16; i <<= 1) {
        const uint32_t t = __shfl_up_sync(mask, val, i, 32);
        if (getLaneId() >= i) val += t;
    }

    const uint32_t t = __shfl_up_sync(mask, val, 1, 32);
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

__global__ void RadixScan(
    uint32_t* passHist,
    const uint32_t threadBlocks
) {
    // const uint32_t stride = blockDim.x;

    // Dynamically allocated shared memory for scan operations
    extern __shared__ uint32_t s_scan[];

    // Calculate thread indices and offsets
    const uint32_t lane_id = getLaneId();
    const uint32_t circular_lane_shift = (lane_id + 1) & LANE_MASK;
    const uint32_t digit_offset = blockIdx.x * threadBlocks;

    // Calculate end of full partitions
    const uint32_t partitions_end = threadBlocks / blockDim.x * blockDim.x;
    uint32_t reduction = 0;

    // Process full blocks - matching original implementation
    for (uint32_t i = threadIdx.x; i < partitions_end; i += blockDim.x) {
        // Load and perform warp-level scan
        s_scan[threadIdx.x] = passHist[i + digit_offset];
        s_scan[threadIdx.x] = InclusiveWarpScan(s_scan[threadIdx.x]);
        __syncthreads();

        // Inter-warp scan - process last element of each warp
        if (threadIdx.x < (blockDim.x >> LANE_LOG)) {
            s_scan[(threadIdx.x + 1 << LANE_LOG) - 1] = 
                ActiveInclusiveWarpScan(s_scan[(threadIdx.x + 1 << LANE_LOG) - 1]);
        }
        __syncthreads();

        // Write result with circular shift to avoid bank conflicts
        passHist[circular_lane_shift + (i & ~LANE_MASK) + digit_offset] =
            (lane_id != LANE_MASK ? s_scan[threadIdx.x] : 0) +
            (threadIdx.x >= WARP_SIZE ? 
                __shfl_sync(0xffffffff, s_scan[threadIdx.x - 1], 0) : 0) +
            reduction;
        
        // Update reduction for next iteration
        reduction += s_scan[blockDim.x - 1];
        __syncthreads();
    }

    // Handle remaining elements
    uint32_t i = threadIdx.x + partitions_end;
    if (i < threadBlocks) {
        // Same process for remainder
        s_scan[threadIdx.x] = passHist[i + digit_offset];
        s_scan[threadIdx.x] = InclusiveWarpScan(s_scan[threadIdx.x]);
        __syncthreads();
        
        if (threadIdx.x < (blockDim.x >> LANE_LOG)) {
            s_scan[(threadIdx.x + 1 << LANE_LOG) - 1] = 
                ActiveInclusiveWarpScan(s_scan[(threadIdx.x + 1 << LANE_LOG) - 1]);
        }
        __syncthreads();
        
        const uint32_t write_idx = circular_lane_shift + (i & ~LANE_MASK) + digit_offset;
        if (write_idx < threadBlocks + digit_offset) {
            passHist[write_idx] =
                (lane_id != LANE_MASK ? s_scan[threadIdx.x] : 0) +
                (threadIdx.x >= WARP_SIZE ? 
                    __shfl_sync(0xffffffff, s_scan[threadIdx.x - 1], 0) : 0) +
                reduction;
        }
    }

    // // const uint32_t laneId = threadIdx.x & LANE_MASK;        // Thread position within warp
    // // const uint32_t warpId = threadIdx.x >> LANE_LOG;        // Warp number within block
    // // const uint32_t digitOffset = blockIdx.x * threadBlocks; // Offset for current radix digit

    // // Initialize variables for processing full blocks
    // uint32_t reduction = 0; // Running sum from previous iterations
    // // Calculate end point for full block processing
    // const uint32_t fullBlocksEnd = (threadBlocks / stride) * stride;

    // // Process full blocks that can use all threads
    // for(uint32_t i=threadIdx.x; i<fullBlocksEnd; i+=stride) {
    //     // Load data and perform warp-level scan
    //     uint32_t val = passHist[i + digitOffset];
    //     val = InclusiveWarpScan(val, laneId);

    //     s_scan[threadIdx.x] = val;
    //     __syncthreads();

    //     // Perform inter-warp scan using last thread of each warp
    //     if (laneId == LANE_MASK) {
    //         uint32_t warpResult = s_scan[threadIdx.x];
    //         // Scan across warp results
    //         warpResult = InclusiveWarpScan(warpResult, warpId);
    //         s_scan[threadIdx.x] = warpResult;
    //     }
    //     __syncthreads();  // Ensure inter-warp scan is complete

    //     // Calculate final value including previous iterations
    //     uint32_t writeVal = val;
    //     if (warpId > 0) {
    //         // Add result from previous warp
    //         writeVal += s_scan[((warpId - 1) << LANE_LOG) + LANE_MASK];
    //     }
    //     writeVal += reduction;  // Add reduction from previous iterations

    //     // Write result with circular shift to avoid bank conflicts
    //     const uint32_t writeIndex = ((i & ~LANE_MASK) + 
    //         ((laneId + 1) & LANE_MASK)) + digitOffset;
    //     passHist[writeIndex] = writeVal;

    //     // Update reduction for next iteration
    //     reduction += s_scan[stride - 1];
    //     __syncthreads();  // Ensure shared memory is ready for next iteration
    // }

    // // Handle remaining elements (partial block)
    // if (threadIdx.x < (threadBlocks - fullBlocksEnd)) {
    //     uint32_t i = fullBlocksEnd + threadIdx.x;
        
    //     // Perform same scan operation as above but with bounds checking
    //     uint32_t val = passHist[i + digitOffset];
    //     val = InclusiveWarpScan(val, laneId);
    //     s_scan[threadIdx.x] = val;
    //     __syncthreads();
        
    //     if (laneId == LANE_MASK) {
    //         uint32_t warpResult = s_scan[threadIdx.x];
    //         warpResult = InclusiveWarpScan(warpResult, warpId);
    //         s_scan[threadIdx.x] = warpResult;
    //     }
    //     __syncthreads();
        
    //     uint32_t writeVal = val;
    //     if (warpId > 0) {
    //         writeVal += s_scan[((warpId - 1) << LANE_LOG) + LANE_MASK];
    //     }
    //     writeVal += reduction;
        
    //     // Write result with bounds checking for partial block
    //     const uint32_t writeIndex = ((i & ~LANE_MASK) + 
    //         ((laneId + 1) & LANE_MASK)) + digitOffset;
    //     if (writeIndex < threadBlocks + digitOffset) {
    //         passHist[writeIndex] = writeVal;
    //     }
    // }
}