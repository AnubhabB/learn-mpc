R"(#include <metal_stdlib>

#define RADIX 256
#define SIMD_SIZE 32 // This can change, can we visit this later?
#define LANE_LOG  5
#define RADIX_LOG 8


#define LANE_MASK (SIMD_SIZE - 1)
#define RADIX_MASK (RADIX - 1)

#define VECTORIZE_SIZE 4

using namespace metal;

/***********************
*
SIMD Group helper functions
*
***********************/
// Scan with circular shift
inline uint32_t inclusive_scan_circular(
    uint32_t val,
    uint thread_index [[thread_index_in_simdgroup]]
) {
    // Do the inclusive scan efficiently using built-in function
    uint32_t scan_result = simd_prefix_inclusive_sum(val);
    
    // Perform the circular shift
    return simd_shuffle(scan_result, (thread_index + LANE_MASK) & LANE_MASK);
}

inline uint32_t active_exclusive_simd_scan(
    uint32_t val,
    uint thread_index [[thread_index_in_simdgroup]]
) {
    // Do inclusive scan
    uint32_t scan_result = simd_prefix_inclusive_sum(val);
    
    // Shift to make it exclusive
    return thread_index > 0 ? simd_shuffle_up(scan_result, 1) : 0;
}

/***********************
*
Helper for Bit conversions
*
***********************/
// Default template for non-floating point types
template<typename T, typename U>
inline U toBits(T val) {
    return static_cast<U>(val);
}

// Explicit specialization for uint32_t
template<>
inline uchar toBits<uint8_t, uint8_t>(uint8_t val) {
    return static_cast<uint8_t>(val);
}

// Explicit specialization for float
template<>
inline uint32_t toBits<float, uint32_t>(float val) {
    if (metal::isfinite(val)) {
        uint32_t bits = as_type<uint32_t>(val);  // Metal's bit casting
        return (bits & 0x80000000) ? ~bits : bits ^ 0x80000000;
    }
    return metal::isnan(val) || val > 0.0f ? 0xFFFFFFFF : 0;
}

// Explicit specialization for uint32_t
template<>
inline uint32_t toBits<uint32_t, uint32_t>(uint32_t val) {
    return static_cast<uint32_t>(val);
}

/***********************
*
Helper for Vectorized load
*
***********************/

// Unified container that always stores the converted type U
template<typename U>
struct Vectorized {
    U x, y, z, w;
};

// Base declaration with both input type T and output type U
template<typename T, typename U>
struct VectorLoad {
    static Vectorized<U> load(const device T* data, uint32_t idx);
};

// Specialization for uint32_t
template<>
struct VectorLoad<uint32_t, uint32_t> {
    static Vectorized<uint32_t> load(const device uint32_t* data, uint32_t idx) {
        // Create aligned pointer to starting address
        const device uint32_t* aligned_ptr = data + idx;
        // Do vectorized load using Metal's vector type
        uint4 vec = *reinterpret_cast<const device uint4*>(aligned_ptr);
        
        return Vectorized<uint32_t>{
            toBits<uint32_t, uint32_t>(vec.x),
            toBits<uint32_t, uint32_t>(vec.y),
            toBits<uint32_t, uint32_t>(vec.z),
            toBits<uint32_t, uint32_t>(vec.w)
        };
    }
};

// Specialization for float
template<>
struct VectorLoad<float, uint32_t> {
    static Vectorized<uint32_t> load(const device float* data, uint32_t idx) {
        // Create aligned pointer to starting address
        const device float* aligned_ptr = data + idx;
        // Do vectorized load using Metal's vector type
        float4 vec = *reinterpret_cast<const device float4*>(aligned_ptr);
        
        return Vectorized<uint32_t>{
            toBits<float, uint32_t>(vec.x),
            toBits<float, uint32_t>(vec.y),
            toBits<float, uint32_t>(vec.z),
            toBits<float, uint32_t>(vec.w)
        };
    }
};

// Specialization for uint8_t
template<>
struct VectorLoad<uint8_t, uint8_t> {
    static Vectorized<uint8_t> load(const device uint8_t* data, uint32_t idx) {
        // Create aligned pointer to starting address
        const device uint8_t* aligned_ptr = data + idx;
        // Do vectorized load using Metal's vector type
        uchar4 vec = *reinterpret_cast<const device uchar4*>(aligned_ptr);
        
        return Vectorized<uint8_t>{
            toBits<uint8_t, uint8_t>(vec.x),
            toBits<uint8_t, uint8_t>(vec.y),
            toBits<uint8_t, uint8_t>(vec.z),
            toBits<uint8_t, uint8_t>(vec.w)
        };
    }
};

// Radix Upsweep pass does the following:
// radixShift - signifies which `digit` position is being worked on in strides of 8 - first pass for MSB -> last 8 bits using radix 256
// passHist - for a particular digit position creates a frequency of values -
// in this implementaiton a passHist is computer per threadBlock and each threadBlock is responsible for processing `numElementsInBlock`
// globalHist - converts these frequencies into cumulative counts (prefix sums)
template<typename T, typename U>
METAL_FUNC void RadixUpsweep(
    device const T* keys,
    device atomic_uint* globalHist,
    device uint32_t* passHist,
    constant uint32_t &size,
    constant uint32_t &radixShift,
    constant uint32_t &partSize,
    threadgroup atomic_uint* s_globalHist [[threadgroup(0)]],
    uint threadIdx [[thread_position_in_threadgroup]],
    uint threadIdxInSimdGroup [[thread_index_in_simdgroup]],
    uint groupIdx [[threadgroup_position_in_grid]],
    uint groupDim [[threads_per_threadgroup]],
    uint gridDim  [[threadgroups_per_grid]]
) {
    // Clear shared memory histogram
    #pragma unroll
    for (uint32_t i = threadIdx; i < RADIX; i += groupDim)
        atomic_store_explicit(&s_globalHist[i], 0u, memory_order_relaxed);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Calculate this block's range
    const uint32_t block_start = groupIdx * partSize;
    const uint32_t block_end = min(block_start + partSize, size);
    const uint32_t elements_in_block = block_end - block_start;

    // Calculate number of full vectors - we are going to make an attempt to process
    const uint32_t full_vecs = elements_in_block / VECTORIZE_SIZE;
    const uint32_t vec_end = block_start + (full_vecs * VECTORIZE_SIZE);

    for (uint32_t i = threadIdx; i < full_vecs; i += groupDim) {
        const uint32_t idx = block_start + i * VECTORIZE_SIZE;

        if (idx < vec_end) {
            Vectorized<U> data = VectorLoad<T, U>::load(keys, idx);
            
            atomic_fetch_add_explicit(&s_globalHist[data.x >> radixShift & RADIX_MASK], 1u, memory_order_relaxed);
            atomic_fetch_add_explicit(&s_globalHist[data.y >> radixShift & RADIX_MASK], 1u, memory_order_relaxed);
            atomic_fetch_add_explicit(&s_globalHist[data.z >> radixShift & RADIX_MASK], 1u, memory_order_relaxed);
            atomic_fetch_add_explicit(&s_globalHist[data.w >> radixShift & RADIX_MASK], 1u, memory_order_relaxed);
        }
    }
    
    // Process remaining elements
    for (uint32_t i = threadIdx + vec_end; i < block_end; i += groupDim) {
        U bits = toBits<T, U>(keys[i]);
        atomic_fetch_add_explicit(&s_globalHist[bits >> radixShift & RADIX_MASK], 1u, memory_order_relaxed);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // threadgroup uint32_t* s_globalHist_recast = reinterpret_cast<threadgroup uint32_t*>(s_globalHist);
    
    // Reduce histograms and prepare for prefix sum
    for (uint32_t i = threadIdx; i < RADIX; i += groupDim) {
        // Memory layout: digit frequencies across all blocks
        // So, if we have n blocks we'll have frequency values for a digit in each blocks consecutively
        uint32_t digitCount = atomic_load_explicit(&s_globalHist[i], memory_order_relaxed);
        // uint32_t digitCount = s_globalHist_recast[i];
        passHist[i * gridDim + groupIdx] = digitCount;
        atomic_store_explicit(&s_globalHist[i], inclusive_scan_circular(digitCount, threadIdxInSimdGroup), memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Atomic feth is bloody costly
    threadgroup uint32_t* s_globalHist_recast = reinterpret_cast<threadgroup uint32_t*>(s_globalHist);

    // Perform warp-level scan - for first thread in each warp
    if (threadIdx < (RADIX >> LANE_LOG))
        s_globalHist_recast[threadIdx << LANE_LOG] = active_exclusive_simd_scan(s_globalHist_recast[threadIdx << LANE_LOG], threadIdxInSimdGroup);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Update global histogram with prefix sum results
    for (uint32_t i = threadIdx; i < RADIX; i += groupDim) {
        atomic_fetch_add_explicit(
            &globalHist[i + (radixShift << LANE_LOG)],
            s_globalHist_recast[i] +
            (
                threadIdxInSimdGroup > 0 ? simd_shuffle(s_globalHist_recast[i - 1], 1) : 0
            ),
            memory_order_relaxed
        );
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
}

#define UPSWEEP(T, U, name)                                   \
kernel void name##_##T##_##U(                                 \
    device const T* keys,                                     \
    device atomic_uint* globalHist,                           \
    device uint32_t* passHist,                                \
    constant uint32_t &size,                                  \
    constant uint32_t &radixShift,                            \
    constant uint32_t &partSize,                              \
    threadgroup atomic_uint* s_globalHist [[threadgroup(0)]], \
    uint threadIdx[[thread_position_in_threadgroup]],         \
    uint threadIdxInSimdGroup [[thread_index_in_simdgroup]],  \
    uint groupIdx[[threadgroup_position_in_grid]],            \
    uint groupDim[[threads_per_threadgroup]],                 \
    uint gridDim  [[threadgroups_per_grid]]                   \
) {                                                           \
    RadixUpsweep<T, U>(keys, globalHist, passHist, size, radixShift, partSize, s_globalHist, threadIdx, threadIdxInSimdGroup, groupIdx, groupDim, gridDim); \
}                                                             \

kernel void RadixScan(
    device uint32_t* passHist,
    constant uint32_t &numPartitions
) {}

kernel void RadixDownsweep() {

}

UPSWEEP(uint8_t, uint8_t, RadixUpsweep)
UPSWEEP(float, uint32_t, RadixUpsweep)
UPSWEEP(uint32_t, uint32_t, RadixUpsweep)
)"