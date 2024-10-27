#include <stdint.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define RADIX               256             //Number of digit bins
#define WARP_SIZE           32
#define LANE_LOG            5               // LANE_LOG = 5 since 2^5 = 32 = warp size
#define RADIX_MASK          (RADIX - 1)     //Mask of digit bins, to extract digits
#define LANE_MASK           (WARP_SIZE - 1)

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
template<typename T> struct VectorType;

// Specializations for different types
template<>
struct VectorType<float> {
    using type = float4;
};

template<>
struct VectorType<uint32_t> {
    using type = uint4;
};

// template<>
// struct VectorType<half> {
    // using type = half4
    // alignas(8) struct {
    //     half x, y, z, w;
    // };
// };

// template<>
// struct VectorType<nv_bfloat16> {
//     using type = alignas(8) struct {
//         nv_bfloat16 x, y, z, w;
//     };
// };

// template<>
// struct VectorType<uint16_t> {
//     using type = alignas(8) struct {
//         uint16_t x, y, z, w;
//     };
// };

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
    using VecT = typename VectorType<T>::type;
    constexpr uint32_t vec_size = 4;

    // Calculate number of full vectors - we are going to make an attempt to process 4 vectors at a time
    const uint32_t full_vecs = elements_in_block / vec_size;

    for (uint32_t i = threadIdx.x; i < full_vecs; i += blockDim.x) {
        const uint32_t idx = block_start / vec_size + i;
        const VecT vec_val = reinterpret_cast<const VecT*>(sort)[idx];
        
        uint32_t values[vec_size] = {
            toBits(vec_val.x),
            toBits(vec_val.y),
            toBits(vec_val.z),
            toBits(vec_val.w)
        };
        
        #pragma unroll
        for (int j = 0; j < vec_size; j++) {
            atomicAdd(&s_globalHist[values[j] >> radixShift & RADIX_MASK], 1);
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