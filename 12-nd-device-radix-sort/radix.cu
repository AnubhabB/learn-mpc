#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define RADIX               256     //Number of digit bins
#define RADIX_MASK          (RADIX - 1)     //Mask of digit bins, to extract digits
#define RADIX_LOG           8

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

template<>
struct VectorType<half> {
    // using type = half4
    // alignas(8) struct {
    //     half x, y, z, w;
    // };
};

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
    __shared__ uint32_t s_globalHist[RADIX * 2];

    // Clear shared memory histogram
    #pragma unroll
    for (uint32_t i = threadIdx.x; i < RADIX * 2; i += blockDim.x)
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
    // const uint32_t partials  = elements_in_block % vec_size;

    for (uint32_t i = threadIdx.x; i < full_vecs; i += blockDim.x) {
        const uint32_t idx = block_start / vec_size + i;
        const VecT vec_val = reinterpret_cast<const VecT*>(sort)[idx];
        
        uint32_t values[4] = {
            toBits(vec_val.x),
            toBits(vec_val.y),
            toBits(vec_val.z),
            toBits(vec_val.w)
        };
        
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            atomicAdd(&s_globalHist[values[j] >> radixShift & RADIX_MASK], 1);
        }
    }

    // Process remaining elements
    const uint32_t vec_end = block_start + (full_vecs * vec_size);
    for (uint32_t i = threadIdx.x + vec_end; i < block_end; i += blockDim.x) {
        const T t = sort[i];
        uint32_t bits = toBits(t, radixShift);
        atomicAdd(&s_globalHist[bits & RADIX_MASK], 1);
    }

    __syncthreads();
}