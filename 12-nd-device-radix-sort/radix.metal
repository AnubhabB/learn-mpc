R"(#include <metal_stdlib>

#define RADIX 256
#define SIMD_SIZE 32 // This can change, can we visit this later?
#define LANE_LOG  5
#define RADIX_LOG 8


#define LANE_MASK (SIMD_SIZE - 1)
#define RADIX_MASK (RADIX - 1)

#define VECTORIZE_SIZE 4
#define BIN_KEYS_PER_THREAD 15

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
    if (isfinite(val)) {
        uint32_t bits = as_type<uint32_t>(val);  // Metal's bit casting
        return (bits & 0x80000000) ? ~bits : bits ^ 0x80000000;
    }

    return isnan(val) || val > 0.0 ? 0xFFFFFFFF : 0;
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

/***********************
*
Helper function to get type-specific maximum value
*
***********************/
template<typename T>
inline T getTypeMax() {
    // this should be unreachable
    return static_cast<T>(1);
}

template<>
inline uint8_t getTypeMax() {
    return 0xFF; // 255
}

template<>
inline float getTypeMax() {
    return INFINITY;
}

template<>
inline uint32_t getTypeMax() {
    return 0xFFFFFFFF;  // 4294967295
}

/***********************
*
Radix sort kernels
*
***********************/

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
}

kernel void RadixScan(
    device uint32_t* passHist,
    constant uint32_t &numPartitions,
    threadgroup uint32_t* s_scan,
    uint threadIdx [[thread_position_in_threadgroup]],
    uint laneIdx   [[thread_index_in_simdgroup]],
    uint groupIdx  [[threadgroup_position_in_grid]],
    uint groupDim  [[threads_per_threadgroup]]
) {
    // Circular shift within warp - this helps reduce bank conflicts
    // Get ID of the next thread: laneIdx: 0 -> 1, 1 -> 2 ... 31 -> 0
    const uint32_t circularLaneShift = (laneIdx + 1) & LANE_MASK;

    // Where does the digit start
    const uint32_t digitOffset = groupIdx * numPartitions;

    // Calculate the number of full block-sized chunks we need to process
    const uint32_t fullBlocksEnd = (numPartitions / groupDim) * groupDim;

    // Running sum for carrying over between iterations
    uint32_t reduction = 0;

    uint32_t tidx = threadIdx;
    for (; tidx < fullBlocksEnd; tidx += groupDim) {
        s_scan[threadIdx] = passHist[threadIdx + digitOffset];

        // Perform warp-level scan
        s_scan[threadIdx] = simd_prefix_inclusive_sum(s_scan[threadIdx]);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Collect and scan warp totals
        if (threadIdx < (groupDim >> LANE_LOG)) {
            s_scan[((threadIdx + 1) << LANE_LOG) - 1] = simd_prefix_inclusive_sum(s_scan[((threadIdx + 1) << LANE_LOG) - 1]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint32_t writeIdx = circularLaneShift + (tidx & ~LANE_MASK);

        passHist[writeIdx + digitOffset] =
            (laneIdx != LANE_MASK ? s_scan[threadIdx] : 0) +
            (threadIdx >= SIMD_SIZE ?
            s_scan[(threadIdx & ~LANE_MASK) - 1] : 0) +
            reduction;

        reduction += s_scan[groupDim - 1];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Remaining elements handled similarly...
    if (tidx < numPartitions) {
        s_scan[threadIdx] = passHist[threadIdx + digitOffset];
    }

    s_scan[threadIdx] = simd_prefix_inclusive_sum(s_scan[threadIdx]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (threadIdx < (groupDim >> LANE_LOG)) {
        s_scan[((threadIdx + 1) << LANE_LOG) - 1] = simd_prefix_inclusive_sum(s_scan[((threadIdx + 1) << LANE_LOG) - 1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint32_t writeIdx = circularLaneShift + (tidx & ~LANE_MASK);
    if (writeIdx < numPartitions) {
        passHist[writeIdx + digitOffset] =
            (laneIdx != LANE_MASK ? s_scan[threadIdx] : 0) +
            (threadIdx >= SIMD_SIZE ?
            s_scan[(threadIdx & ~LANE_MASK) - 1] : 0) +
            reduction;
    }
}

template<typename T, typename U>
METAL_FUNC void RadixDownsweep(
    device const T* keys,               // Input array
    device T* keysAlt,                  // Output array
    device const uint32_t* vals,        // [Optional] payload/ values to be sorted - for our usecase these are indices
    device uint32_t* valsAlt,           // [Optional] output for values/ payload
    device const uint32_t* globalHist,  // Global histogram
    device const uint32_t* passHist,    // Pass histogram
    constant uint32_t &size,            // length of input array
    constant uint32_t &radixShift,      // current radixShift bit
    constant uint32_t &partSize,        // Number of elements processed per partition/ block
    constant uint32_t &histSize,        // size of the histogram initialized externally
    constant uint32_t &numKeysPerThread,// real number of keys processed by each thread. Max would be `BIN_KEYS_PER_THREAD`
    constant bool     &sortIdx,         // if set to true, attempt to sort the indices
    threadgroup uint32_t* s_tmp [[threadgroup(0)]],            // Shared memory layout, used for `s_simdHistograms` and later for `s_keys` and [optional]`s_values`
    threadgroup uint32_t* s_localHistogram [[threadgroup(1)]], // local histogram/ secondary storage
    uint threadIdx [[thread_position_in_threadgroup]],
    uint laneIdx   [[thread_index_in_simdgroup]],
    uint simdIdx   [[simdgroup_index_in_threadgroup]],
    uint groupIdx  [[threadgroup_index_in_grid]],
    uint groupDim  [[threads_per_threadgroup]],
    uint gridDim   [[threadgroups_per_grid]]
) {
    volatile threadgroup atomic_uint* s_simdHist = reinterpret_cast<threadgroup atomic_uint*>(&s_tmp[simdIdx << RADIX_LOG]);

    // for warp histogram temp storage
    threadgroup uint32_t* s_simdHistograms = s_tmp;

    // The partition offset of keys to work with
    const uint32_t blockOffset = groupIdx * partSize;
    const uint32_t tidInSimd   = ((numKeysPerThread << LANE_LOG) * simdIdx) + laneIdx;

    //clear shared memory
    for (uint32_t i = threadIdx; i < histSize; i += groupDim)
        s_tmp[i] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    thread uint32_t threadStore[BIN_KEYS_PER_THREAD]; // local store for max keys per thread to be later used for indices
    thread uint16_t offsets[BIN_KEYS_PER_THREAD];

    thread T* threadKeys = reinterpret_cast<thread T*>(threadStore);

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
    for (uint32_t i=0, t=blockOffset + tidInSimd; i<numKeysPerThread;++i, t+=SIMD_SIZE) {
        threadKeys[i] = t < size ? keys[t] : getTypeMax<T>();
    }
    threadgroup_barrier(mem_flags::mem_device);

    // WLMS (warp-level multi-split) Ashkiani et al (https://arxiv.org/pdf/1701.01189)
    // Computes warp level histogram for digits
    #pragma unroll
    for (uint32_t i = 0; i < numKeysPerThread; ++i) {
        U bitval = toBits<T, U>(threadKeys[i]);

        // creating mask for threads in a warp that have same bit value as keys[i]
        unsigned simdFlags = 0xffffffff;
        #pragma unroll
        for (int k = 0; k < RADIX_LOG; ++k) {
            // true if `radixShift + kth` position is 1
            const bool t2 = (bitval >> (k + radixShift)) & 1;
            simdFlags &= (t2 ? 0 : 0xffffffff) ^ (simd_vote::vote_t)simd_ballot(t2);
        }

        const uint32_t bits = popcount(simdFlags & ((1u << laneIdx) - 1));
        uint32_t preIncrementVal;
        // Update histogram count only once per warp
        if (bits == 0) {
            preIncrementVal = atomic_fetch_add_explicit(&s_simdHist[(bitval >> radixShift) & RADIX_MASK], popcount(simdFlags), memory_order_relaxed);
        }

        offsets[i] = simd_shuffle(preIncrementVal, ctz(simdFlags)) + bits;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // At this stage we have `simd-level` histograms
    // exclusive prefix sum up the simd histograms
    if (threadIdx < RADIX) {
        uint32_t reduction = s_simdHistograms[threadIdx];
        for (uint32_t i = threadIdx + RADIX; i < histSize; i += RADIX) {
            reduction += s_simdHistograms[i];
            s_simdHistograms[i] = reduction - s_simdHistograms[i];
        }

        //begin the exclusive prefix sum across the reductions
        s_simdHistograms[threadIdx] = inclusive_scan_circular(reduction, laneIdx);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Update the first threads of warps
    if (threadIdx < (RADIX >> LANE_LOG)) {
        uint32_t val = s_simdHistograms[threadIdx << LANE_LOG];
        s_simdHistograms[threadIdx << LANE_LOG] = active_exclusive_simd_scan(val, laneIdx);

    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (threadIdx < RADIX && laneIdx)
        s_simdHistograms[threadIdx] += simd_shuffle(s_simdHistograms[threadIdx - 1], 1);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    //update offsets
    if (simdIdx) {
        #pragma unroll
        for (uint32_t i = 0; i < numKeysPerThread; ++i) {
            const U t2 = toBits<T, U>(threadKeys[i]) >> radixShift & RADIX_MASK;
            offsets[i] += static_cast<uint32_t>(atomic_load_explicit(&s_simdHist[t2], memory_order_relaxed)) + s_simdHistograms[t2];
        }
    } else {
        #pragma unroll
        for (uint32_t i = 0; i < numKeysPerThread; ++i)
            offsets[i] += s_simdHistograms[toBits<T, U>(threadKeys[i]) >> radixShift & RADIX_MASK];
    }

    //load in threadblock reductions
    #pragma unroll
    for (uint32_t i=threadIdx; i<RADIX; i+=groupDim) {
        s_localHistogram[i] = globalHist[i + (radixShift << LANE_LOG)] +
            passHist[i * gridDim + groupIdx] - s_simdHistograms[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // `s_tmp` has done with it's job as warp histogram bookkeeper
    // let's re-use it for our keys
    threadgroup T* s_keys = reinterpret_cast<threadgroup T*>(s_tmp);
    
    // scatter keys into shared memory
    #pragma unroll
    for (uint32_t i = 0; i < numKeysPerThread; ++i) {
        s_keys[offsets[i]] = threadKeys[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    //scatter runs of keys into device memory
    uint8_t digits[BIN_KEYS_PER_THREAD];
    uint32_t pSize = size - blockOffset;
    #pragma unroll
    for(uint32_t i=0, t=threadIdx; i<BIN_KEYS_PER_THREAD; ++i, t += groupDim) {
        if (i < numKeysPerThread && t < pSize) {
            digits[i] = toBits<T, U>(s_keys[t]) >> radixShift & RADIX_MASK;
            keysAlt[s_localHistogram[digits[i]] + t] = s_keys[t];
        }
    }

    if (!sortIdx)
        return;
    threadgroup_barrier(mem_flags::mem_device); // this is required only if we proceed with sorting of indices

    /*
    // `s_tmp` has done with it's job as warp histogram bookkeeper & keys
    // let's re-use it for our vals
    threadgroup uint32_t* s_vals = reinterpret_cast<threadgroup uint32_t*>(s_tmp);
    thread uint32_t* threadVals = reinterpret_cast<thread uint32_t*>(threadStore);

    // Load indices into registers
    #pragma unroll
    for (uint32_t i = 0, t = blockOffset + tidInSimd; i < numKeysPerThread; ++i, t += SIMD_SIZE) {
        threadVals[i] = t < size ? vals[t] : size; // `size` is a placeholder here > max index
    }
    threadgroup_barrier(mem_flags::mem_device);

    // scatter keys into shared memory
    #pragma unroll
    for (uint32_t i = 0; i < numKeysPerThread; ++i) {
        s_vals[offsets[i]] = threadVals[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma unroll
    for(uint32_t i=0, t=threadIdx; i<BIN_KEYS_PER_THREAD; ++i, t += groupDim) {
        if (i < numKeysPerThread && t < pSize) {
            valsAlt[s_localHistogram[digits[i]] + t] = s_vals[t];
        }
    }
    */
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

#define DOWNSWEEP(T, U, name)                                 \
kernel void name##_##T##_##U(                                 \
    device const T* keys,                                     \
    device T* keysAlt,                                        \
    device const uint32_t* vals,                              \
    device uint32_t* valsAlt,                                 \
    device const uint32_t* globalHist,                        \
    device const uint32_t* passHist,                          \
    constant uint32_t &size,                                  \
    constant uint32_t &radixShift,                            \
    constant uint32_t &partSize,                              \
    constant uint32_t &histSize,                              \
    constant uint32_t &numKeysPerThread,                      \
    constant bool     &sortIdx,                               \
    threadgroup uint32_t* s_tmp [[threadgroup(0)]],           \
    threadgroup uint32_t* s_localHistogram [[threadgroup(1)]],\
    uint threadIdx [[thread_position_in_threadgroup]],        \
    uint laneIdx   [[thread_index_in_simdgroup]],             \
    uint simdIdx   [[simdgroup_index_in_threadgroup]],        \
    uint groupIdx  [[threadgroup_position_in_grid]],          \
    uint groupDim  [[threads_per_threadgroup]],               \
    uint gridDim  [[threadgroups_per_grid]]                   \
) {                                                           \
    RadixDownsweep<T, U>(                                     \
        keys,                                                 \
        keysAlt,                                              \
        vals,                                                 \
        valsAlt,                                              \
        globalHist,                                           \
        passHist,                                             \
        size,                                                 \
        radixShift,                                           \
        partSize,                                             \
        histSize,                                             \
        numKeysPerThread,                                     \
        sortIdx,                                              \
        s_tmp,                                                \
        s_localHistogram,                                     \
        threadIdx,                                            \
        laneIdx,                                              \
        simdIdx,                                              \
        groupIdx,                                             \
        groupDim,                                             \
        gridDim                                               \
    );                                                        \
}                                                             \

/***********************
*
Macros
*
***********************/

UPSWEEP(uint8_t, uint8_t, RadixUpsweep)
UPSWEEP(float, uint32_t, RadixUpsweep)
UPSWEEP(uint32_t, uint32_t, RadixUpsweep)

DOWNSWEEP(uint8_t, uint8_t, RadixDownsweep)
DOWNSWEEP(float, uint32_t, RadixDownsweep)
DOWNSWEEP(uint32_t, uint32_t, RadixDownsweep)
)"