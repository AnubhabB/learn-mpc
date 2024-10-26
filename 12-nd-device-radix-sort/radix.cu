#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Constants that can be adjusted based on device properties
#define MAX_THREADS_PER_BLOCK     1024    // Maximum threads per block
#define WARP_SIZE                 32      // Warp size (constant for current CUDA architectures)
#define MIN_BLOCKS_PER_SM         2       // Minimum blocks per SM for good occupancy
#define DEFAULT_BLOCK_SIZE        256     // Default block size for good occupancy

#define RADIX                     256         //Number of digit bins
#define RADIX_MASK                (RADIX - 1) // Masks of digits to extract
#define RADIX_LOG                 8
#define ITEMS_PER_THREAD          8

#define WARP_SIZE                 32 // current cuda warp size
#define LANE_MASK                 (WARP_SIZE - 1)
#define LANE_LOG                  5 // LANE_LOG = 5 since 2^5 = 32 = warp size

__device__ __forceinline__ uint32_t getLaneId() {
    uint32_t laneId;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
    return laneId;
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

// Prefix sum of active threads ot including itself
__device__ __forceinline__ uint32_t ActiveExclusiveWarpScan(uint32_t val) {
    const uint32_t mask = __activemask();
    #pragma unroll
    for (int i = 1; i <= 16; i <<= 1) // 16 = LANE_COUNT >> 1
    {
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
        return (__float_as_uint(val) ^ ((__float_as_uint(val) >> 31) | 0x80000000));
    } else if constexpr (std::is_same<T, __half>::value) {
        uint16_t bits = __half_as_ushort(val);
        uint16_t mask = -int(bits >> 15) | 0x8000;
        return static_cast<uint32_t>(bits ^ mask);
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        uint16_t bits = __bfloat16_as_ushort(val);
        uint16_t mask = -int(bits >> 15) | 0x8000;
        return static_cast<uint32_t>(bits ^ mask);
    } else if constexpr (std::is_same<T, int64_t>::value) {
        // return static_cast<uint32_t>((val >> radixShift) & 0xFFFFFFFF);
    } else {
        return static_cast<uint32_t>(val);
    }
}

// Cuda kernel for RadixUpsweep
// template<typename T>
// __global__ void RadixUpsweep(
//     T* sort,
//     uint32_t* globalHist,
//     uint32_t* passHist,
//     uint32_t size,
//     uint32_t radixShift,
//     uint32_t partitionSize
// ) {
//     // Shared memory for local histograms
//     // Two histograms per thread block for better bank conflict avoidance
//     __shared__ uint32_t s_globalHist[RADIX * 2];

//     // Clear shared memory histograms
//     for (uint32_t i = threadIdx.x; i < RADIX * 2; i += blockDim.x) {
//         s_globalHist[i] = 0;
//     }
//     __syncthreads();

//     // Calculate this block's work range
//     // Calculate block's work range
//     const uint32_t blockStart = blockIdx.x * partitionSize;
//     const uint32_t blockEnd = min(blockStart + partitionSize, size);

//     // Get pointer to this wave's histogram in shared memory
//     // Using 64 threads per histogram for better parallelism
//     uint32_t* s_wavesHist = &s_globalHist[(threadIdx.x / 64) * RADIX];

//     // Process elements based on type size for vectorized loading
//     constexpr uint32_t typeSize = sizeof(T);
    
//     // float or uint32_t
//     if constexpr (typeSize == 4) {
//         // Vector load optimization for 4-byte types
//         using Vec4T = typename std::conditional<std::is_same<T, float>::value, float4, uint4>::type;
//         // Calculate vector boundaries
//         const uint32_t vecStart = blockStart / 4;
//         const uint32_t vecEnd = (blockEnd + 3) / 4;
//         const Vec4T* vecPtr = reinterpret_cast<const Vec4T*>(sort);

//         // Process full vector elements
//         for (uint32_t i = vecStart + threadIdx.x; i < vecEnd - 1; i += blockDim.x) {
//             const Vec4T vec = vecPtr[i];
            
//             // Convert vector elements to sortable bits
//             const uint32_t x = toBits(vec.x);
//             const uint32_t y = toBits(vec.y);
//             const uint32_t z = toBits(vec.z);
//             const uint32_t w = toBits(vec.w);
            
//             // Update histogram atomically
//             atomicAdd(&s_wavesHist[x >> radixShift & RADIX_MASK], 1);
//             atomicAdd(&s_wavesHist[y >> radixShift & RADIX_MASK], 1);
//             atomicAdd(&s_wavesHist[z >> radixShift & RADIX_MASK], 1);
//             atomicAdd(&s_wavesHist[w >> radixShift & RADIX_MASK], 1);
//         }

//         // Handle remaining elements in last vector
//         if (threadIdx.x < (blockEnd - blockStart) % 4 && 
//             vecStart + threadIdx.x == vecEnd - 1) {
//             const Vec4T vec = vecPtr[vecEnd - 1];
//             const uint32_t remaining = blockEnd - ((vecEnd - 1) * 4);
            
//             if (remaining > 0) {
//                 const uint32_t x = toBits(vec.x);
//                 atomicAdd(&s_wavesHist[x >> radixShift & RADIX_MASK], 1);
//             }
//             if (remaining > 1) {
//                 const uint32_t y = toBits(vec.y);
//                 atomicAdd(&s_wavesHist[y >> radixShift & RADIX_MASK], 1);
//             }
//             if (remaining > 2) {
//                 const uint32_t z = toBits(vec.z);
//                 atomicAdd(&s_wavesHist[z >> radixShift & RADIX_MASK], 1);
//             }
//         }
//     } else if constexpr (typeSize == 2) {
//         // bfloat16 or float16
//         // TODO
//     } else {
//         // unsigned char or other types
//         // Process elements one by one
//         for (uint32_t i = blockStart + threadIdx.x; i < blockEnd; i += blockDim.x) {
//             const T val = sort[i];
//             const uint32_t bits = toBits(val);
//             atomicAdd(&s_wavesHist[bits >> radixShift & RADIX_MASK], 1);
//         }
//     }

//     __syncthreads();

//     // Reduce histograms and prepare for prefix sum
//     for (uint32_t i = threadIdx.x; i < RADIX; i += blockDim.x) {
//         // Combine the two histograms
//         s_globalHist[i] += s_globalHist[i + RADIX];
        
//         // Store block's local histogram
//         passHist[i * gridDim.x + blockIdx.x] = s_globalHist[i];
        
//         // Prepare for prefix sum
//         s_globalHist[i] = InclusiveWarpScanCircularShift(s_globalHist[i]);
//     }
//     __syncthreads();

//     // Perform prefix sum on histogram
//     if (threadIdx.x < (RADIX >> LANE_LOG)) {
//         s_globalHist[threadIdx.x << LANE_LOG] = ActiveExclusiveWarpScan(s_globalHist[threadIdx.x << LANE_LOG]);
//     }
//     __syncthreads();

//     // Update global histogram
//     for (uint32_t i = threadIdx.x; i < RADIX; i += blockDim.x) {
//         const uint32_t laneOffset = (getLaneId() ? 
//             __shfl_sync(0xfffffffe, s_globalHist[i - 1], 1) : 0);
//         atomicAdd(&globalHist[i + (radixShift << LANE_LOG)], s_globalHist[i] + laneOffset);
//     }
// }
template<typename T>
__global__ void RadixUpsweep(
    T* sort,
    uint32_t* globalHist,
    uint32_t* passHist,
    uint32_t size,
    uint32_t radixShift,
    uint32_t partitionSize
) {
    // Shared memory for block-local histogram
    __shared__ uint32_t s_hist[RADIX];
    
    // Initialize shared memory
    for (uint32_t i = threadIdx.x; i < RADIX; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();
    
    // Calculate block's work range
    const uint32_t blockStart = blockIdx.x * partitionSize;
    const uint32_t blockEnd = min(blockStart + partitionSize, size);
    
    // Process elements
    for (uint32_t i = blockStart + threadIdx.x; i < blockEnd; i += blockDim.x) {
        const T val = sort[i];
        const uint32_t bits = toBits(val);
        const uint32_t digit = (bits >> radixShift) & RADIX_MASK;
        atomicAdd(&s_hist[digit], 1);
    }
    __syncthreads();
    
    // Store block-local histogram
    for (uint32_t i = threadIdx.x; i < RADIX; i += blockDim.x) {
        passHist[i * gridDim.x + blockIdx.x] = s_hist[i];
        atomicAdd(&globalHist[i], s_hist[i]);
    }
}