#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define randnum(min, max) \
        ((rand() % (int)(((max) + 1) - (min))) + (min))

#define RADIX               256     //Number of digit bins
#define RADIX_MASK          255     //Mask of digit bins, to extract digits
#define RADIX_LOG           8

#define LANE_COUNT          32
#define LANE_MASK           31
#define LANE_LOG            5
#define WARP_INDEX          (threadIdx.x >> LANE_LOG)

//For the upfront global histogram kernel
#define PART_SIZE			7680
#define VEC_PART_SIZE		1920

//For the digit binning
#define BIN_PART_SIZE       7680                                    //Partition tile size in k_DigitBinning
#define BIN_HISTS_SIZE      4096                                    //Total size of warp histograms in shared memory in k_DigitBinning
#define BIN_SUB_PART_SIZE   480                                     //Subpartition tile size of a single warp in k_DigitBinning
#define BIN_KEYS_PER_THREAD 15                                      //Keys per thread in k_DigitBinning
#define BIN_SUB_PART_START  (WARP_INDEX * BIN_SUB_PART_SIZE)        //Starting offset of a subpartition tile
#define BIN_PART_START      (blockIdx.x * BIN_PART_SIZE)			//Starting offset of a partition tile

__device__ __forceinline__ uint32_t getLaneId() 
{
    uint32_t laneId;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
    return laneId;
}

__device__ __forceinline__ unsigned getLaneMaskLt() 
{
    unsigned mask;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
    return mask;
}

__device__ __forceinline__ uint32_t InclusiveWarpScanCircularShift(uint32_t val)
{
    #pragma unroll
    for (int i = 1; i <= 16; i <<= 1) // 16 = LANE_COUNT >> 1
    {
        const uint32_t t = __shfl_up_sync(0xffffffff, val, i, 32);
        if (getLaneId() >= i) val += t;
    }

    return __shfl_sync(0xffffffff, val, getLaneId() + LANE_MASK & LANE_MASK);
}

__device__ __forceinline__ uint32_t ActiveExclusiveWarpScan(uint32_t val)
{
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

//Warp scans
__device__ __forceinline__ uint32_t InclusiveWarpScan(uint32_t val)
{
    #pragma unroll
    for (int i = 1; i <= 16; i <<= 1) // 16 = LANE_COUNT >> 1
    {
        const uint32_t t = __shfl_up_sync(0xffffffff, val, i, 32);
        if (getLaneId() >= i) val += t;
    }

    return val;
}

__device__ __forceinline__ uint32_t ActiveInclusiveWarpScan(uint32_t val)
{
    const uint32_t mask = __activemask();
    #pragma unroll
    for (int i = 1; i <= 16; i <<= 1)
    {
        const uint32_t t = __shfl_up_sync(mask, val, i, 32);
        if (getLaneId() >= i) val += t;
    }

    return val;
}

// Helper functions for bit conversions
template<typename T>
__device__ inline uint32_t toBits(T val, uint32_t radixShift = 0) {
    if constexpr (std::is_same<T, float>::value) {
        uint32_t bits;
        memcpy(&bits, &val, sizeof(float));
        uint32_t mask = -int(bits >> 31) | 0x80000000;
        return bits ^ mask;
    }
    else if constexpr (std::is_same<T, __half>::value) {
        uint16_t bits;
        memcpy(&bits, &val, sizeof(__half));
        uint16_t mask = -int(bits >> 15) | 0x8000;
        return static_cast<uint32_t>(bits ^ mask);
    }
    else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        uint16_t bits;
        memcpy(&bits, &val, sizeof(__nv_bfloat16));
        uint16_t mask = -int(bits >> 15) | 0x8000;
        return static_cast<uint32_t>(bits ^ mask);
    }
    else if constexpr (std::is_same<T, int64_t>::value) {
        // For 64-bit values, we need to handle the radixShift differently
        return static_cast<uint32_t>((val >> radixShift) & 0xFFFFFFFF);
    }
    else {
        // For integral types, just cast
        return static_cast<uint32_t>(val);
    }
}

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

template<typename T>
__global__ void RadixUpsweep(
    T* sort,
    uint32_t* globalHist,
    uint32_t* passHist,
    uint32_t size,
    uint32_t radixShift
) {
    __shared__ uint32_t s_globalHist[RADIX * 2];

    //clear shared memory
    for (uint32_t i = threadIdx.x; i < RADIX * 2; i += blockDim.x)
        s_globalHist[i] = 0;
    __syncthreads();
    
    //histogram
    {
        //64 threads : 1 histogram in shared memory
        uint32_t* s_wavesHist = &s_globalHist[threadIdx.x / 64 * RADIX];

        if (blockIdx.x < gridDim.x - 1)
        {
            const uint32_t partEnd = (blockIdx.x + 1) * VEC_PART_SIZE;
            
            // Vector load based on types
            constexpr uint32_t typesize = sizeof(T);

            // For uint32_t, float (maybe int32??)
            if(typesize == 4) {
                using VecT = typename std::conditional<std::is_same<T, float>::value, 
                                                         float4, uint4>::type;

                for (uint32_t i = threadIdx.x + (blockIdx.x * VEC_PART_SIZE); i < partEnd; i += blockDim.x) {
                    const VecT t = reinterpret_cast<VecT*>(sort)[i];
                    // Convert to sortable bits based on type
                    uint32_t x = toBits(t.x);
                    uint32_t y = toBits(t.y);
                    uint32_t z = toBits(t.z);
                    uint32_t w = toBits(t.w);

                    if(i < 5) {
                        printf("First block: [%u %u %u %u]\n", x, y, z, w);
                    }
                    atomicAdd(&s_wavesHist[x >> radixShift & RADIX_MASK], 1);
                    atomicAdd(&s_wavesHist[y >> radixShift & RADIX_MASK], 1);
                    atomicAdd(&s_wavesHist[z >> radixShift & RADIX_MASK], 1);
                    atomicAdd(&s_wavesHist[w >> radixShift & RADIX_MASK], 1);
                }
            }
        }

        if (blockIdx.x == gridDim.x - 1)
        {
            for (uint32_t i = threadIdx.x + (blockIdx.x * PART_SIZE); i < size; i += blockDim.x)
            {
                const T t = sort[i];
                uint32_t bits = toBits(t, radixShift);
                if(i < 5) {
                    printf("Second block: [%u %u]\n", i, bits);
                }
                atomicAdd(&s_wavesHist[bits >> radixShift & RADIX_MASK], 1);
            }
        }
    }
    __syncthreads();

    //reduce to the first hist, pass out, begin prefix sum
    for (uint32_t i = threadIdx.x; i < RADIX; i += blockDim.x)
    {
        s_globalHist[i] += s_globalHist[i + RADIX];
        passHist[i * gridDim.x + blockIdx.x] = s_globalHist[i];
        s_globalHist[i] = InclusiveWarpScanCircularShift(s_globalHist[i]);
    }	
    __syncthreads();

    if (threadIdx.x < (RADIX >> LANE_LOG))
        s_globalHist[threadIdx.x << LANE_LOG] = ActiveExclusiveWarpScan(s_globalHist[threadIdx.x << LANE_LOG]);
    __syncthreads();
    
    //Atomically add to device memory
    for (uint32_t i = threadIdx.x; i < RADIX; i += blockDim.x)
        atomicAdd(&globalHist[i + (radixShift << 5)], s_globalHist[i] + (getLaneId() ? __shfl_sync(0xfffffffe, s_globalHist[i - 1], 1) : 0));
}

// __global__ void RadixUpsweep(
//     uint32_t* sort,
//     uint32_t* globalHist,
//     uint32_t* passHist,
//     uint32_t size,
//     uint32_t radixShift
// ) {
//     __shared__ uint32_t s_globalHist[RADIX * 2];

//     //clear shared memory
//     for (uint32_t i = threadIdx.x; i < RADIX * 2; i += blockDim.x)
//         s_globalHist[i] = 0;
//     __syncthreads();
    
//     //histogram
//     {
//         //64 threads : 1 histogram in shared memory
//         uint32_t* s_wavesHist = &s_globalHist[threadIdx.x / 64 * RADIX];

//         if (blockIdx.x < gridDim.x - 1)
//         {
//             const uint32_t partEnd = (blockIdx.x + 1) * VEC_PART_SIZE;
            
//             for (uint32_t i = threadIdx.x + (blockIdx.x * VEC_PART_SIZE); i < partEnd; i += blockDim.x)
//             {
//                 const uint4 t = reinterpret_cast<uint4*>(sort)[i];
//                 if(i < 5) {
//                     printf("First block: [%u %u %u %u]\n", t.x, t.y, t.z, t.w);
//                 }
//                 atomicAdd(&s_wavesHist[t.x >> radixShift & RADIX_MASK], 1);
//                 atomicAdd(&s_wavesHist[t.y >> radixShift & RADIX_MASK], 1);
//                 atomicAdd(&s_wavesHist[t.z >> radixShift & RADIX_MASK], 1);
//                 atomicAdd(&s_wavesHist[t.w >> radixShift & RADIX_MASK], 1);
//             }
//         }

//         if (blockIdx.x == gridDim.x - 1)
//         {
//             for (uint32_t i = threadIdx.x + (blockIdx.x * PART_SIZE); i < size; i += blockDim.x)
//             {
//                 const uint32_t t = sort[i];
//                 if(i < 5) {
//                     printf("Second block: [%u %u]\n", i, t);
//                 }
//                 atomicAdd(&s_wavesHist[t >> radixShift & RADIX_MASK], 1);
//             }
//         }
//     }
//     __syncthreads();

//     //reduce to the first hist, pass out, begin prefix sum
//     for (uint32_t i = threadIdx.x; i < RADIX; i += blockDim.x)
//     {
//         s_globalHist[i] += s_globalHist[i + RADIX];
//         passHist[i * gridDim.x + blockIdx.x] = s_globalHist[i];
//         s_globalHist[i] = InclusiveWarpScanCircularShift(s_globalHist[i]);
//     }	
//     __syncthreads();

//     if (threadIdx.x < (RADIX >> LANE_LOG))
//         s_globalHist[threadIdx.x << LANE_LOG] = ActiveExclusiveWarpScan(s_globalHist[threadIdx.x << LANE_LOG]);
//     __syncthreads();
    
//     //Atomically add to device memory
//     for (uint32_t i = threadIdx.x; i < RADIX; i += blockDim.x)
//         atomicAdd(&globalHist[i + (radixShift << 5)], s_globalHist[i] + (getLaneId() ? __shfl_sync(0xfffffffe, s_globalHist[i - 1], 1) : 0));
// }


__global__ void RadixScan(
    uint32_t* passHist,
    uint32_t threadBlocks)
{
    __shared__ uint32_t s_scan[128];

    uint32_t reduction = 0;
    const uint32_t circularLaneShift = getLaneId() + 1 & LANE_MASK;
    const uint32_t partitionsEnd = threadBlocks / blockDim.x * blockDim.x;
    const uint32_t digitOffset = blockIdx.x * threadBlocks;

    uint32_t i = threadIdx.x;
    for (; i < partitionsEnd; i += blockDim.x)
    {
        s_scan[threadIdx.x] = passHist[i + digitOffset];
        s_scan[threadIdx.x] = InclusiveWarpScan(s_scan[threadIdx.x]);
        __syncthreads();

        if (threadIdx.x < (blockDim.x >> LANE_LOG))
        {
            s_scan[(threadIdx.x + 1 << LANE_LOG) - 1] = 
                ActiveInclusiveWarpScan(s_scan[(threadIdx.x + 1 << LANE_LOG) - 1]);
        }
        __syncthreads();

        passHist[circularLaneShift + (i & ~LANE_MASK) + digitOffset] =
            (getLaneId() != LANE_MASK ? s_scan[threadIdx.x] : 0) +
            (threadIdx.x >= LANE_COUNT ? __shfl_sync(0xffffffff, s_scan[threadIdx.x - 1], 0) : 0) +
            reduction;

        reduction += s_scan[blockDim.x - 1];
        __syncthreads();
    }

    if(i < threadBlocks)
        s_scan[threadIdx.x] = passHist[i + digitOffset];
    s_scan[threadIdx.x] = InclusiveWarpScan(s_scan[threadIdx.x]);
    __syncthreads();

    if (threadIdx.x < (blockDim.x >> LANE_LOG))
    {
        s_scan[(threadIdx.x + 1 << LANE_LOG) - 1] =
            ActiveInclusiveWarpScan(s_scan[(threadIdx.x + 1 << LANE_LOG) - 1]);
    }
    __syncthreads();

    const uint32_t index = circularLaneShift + (i & ~LANE_MASK);
    if (index < threadBlocks)
    {
        passHist[index + digitOffset] =
            (getLaneId() != LANE_MASK ? s_scan[threadIdx.x] : 0) +
            (threadIdx.x >= LANE_COUNT ?
            s_scan[(threadIdx.x & ~LANE_MASK) - 1] : 0) +
            reduction;
    }
}

template<typename T>
__global__ void RadixDownsweepPairs(
    T* sort,
    uint32_t* sortPayload,
    T* alt, 
    uint32_t* altPayload,
    uint32_t* globalHist,
    uint32_t* passHist,
    uint32_t size, 
    uint32_t radixShift)
{
    __shared__ uint32_t s_warpHistograms[BIN_PART_SIZE];
    __shared__ uint32_t s_localHistogram[RADIX];
    volatile uint32_t* s_warpHist = &s_warpHistograms[WARP_INDEX << RADIX_LOG];

    //clear shared memory
    for (uint32_t i = threadIdx.x; i < BIN_HISTS_SIZE; i += blockDim.x)
        s_warpHistograms[i] = 0;

    //load keys
    T keys[BIN_KEYS_PER_THREAD];
    if (blockIdx.x < gridDim.x - 1)
    {
        #pragma unroll
        for (uint32_t i = 0, t = getLaneId() + BIN_SUB_PART_START + BIN_PART_START; i < BIN_KEYS_PER_THREAD; ++i, t += LANE_COUNT)
            keys[i] = sort[t];
    }

    //To handle input sizes not perfect multiples of the partition tile size,
    //load "dummy" keys, which are keys with the highest possible digit.
    //Because of the stability of the sort, these keys are guaranteed to be 
    //last when scattered. This allows for effortless divergence free sorting
    //of the final partition.
    // We'll also incorporate type specific maximum value
    if (blockIdx.x == gridDim.x - 1)
    {
        #pragma unroll
        for (uint32_t i = 0, t = getLaneId() + BIN_SUB_PART_START + BIN_PART_START; i < BIN_KEYS_PER_THREAD; ++i, t += LANE_COUNT)
            keys[i] = t < size ? sort[t] : getTypeMax<T>();
    }
    __syncthreads();

    //WLMS
    uint16_t offsets[BIN_KEYS_PER_THREAD];
    #pragma unroll
    for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
    {
        unsigned warpFlags = 0xffffffff;
        #pragma unroll
        for (int k = 0; k < RADIX_LOG; ++k) {
            // Convert to sortable bits before extracting radix
            uint32_t bits = toBits(keys[i], radixShift);
            const bool t2 = bits >> k + radixShift & 1;
            warpFlags &= (t2 ? 0 : 0xffffffff) ^ __ballot_sync(0xffffffff, t2);
        }
        const uint32_t bits = __popc(warpFlags & getLaneMaskLt());
        uint32_t preIncrementVal;
        if (bits == 0)
            preIncrementVal = atomicAdd((uint32_t*)&s_warpHist[toBits(keys[i]) >> radixShift & RADIX_MASK], __popc(warpFlags));

        offsets[i] = __shfl_sync(0xffffffff, preIncrementVal, __ffs(warpFlags) - 1) + bits;
    }
    __syncthreads();

    //exclusive prefix sum up the warp histograms
    if (threadIdx.x < RADIX)
    {
        uint32_t reduction = s_warpHistograms[threadIdx.x];
        for (uint32_t i = threadIdx.x + RADIX; i < BIN_HISTS_SIZE; i += RADIX)
        {
            reduction += s_warpHistograms[i];
            s_warpHistograms[i] = reduction - s_warpHistograms[i];
        }

        //begin the exclusive prefix sum across the reductions
        s_warpHistograms[threadIdx.x] = InclusiveWarpScanCircularShift(reduction);
    }
    __syncthreads();

    if (threadIdx.x < (RADIX >> LANE_LOG))
        s_warpHistograms[threadIdx.x << LANE_LOG] = ActiveExclusiveWarpScan(s_warpHistograms[threadIdx.x << LANE_LOG]);
    __syncthreads();

    if (threadIdx.x < RADIX && getLaneId())
        s_warpHistograms[threadIdx.x] += __shfl_sync(0xfffffffe, s_warpHistograms[threadIdx.x - 1], 1);
    __syncthreads();

    //update offsets
    if (WARP_INDEX)
    {
        #pragma unroll 
        for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
        {
            const uint32_t t2 = toBits(keys[i]) >> radixShift & RADIX_MASK;
            offsets[i] += s_warpHist[t2] + s_warpHistograms[t2];
        }
    }
    else
    {
        #pragma unroll
        for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
            offsets[i] += s_warpHistograms[toBits(keys[i]) >> radixShift & RADIX_MASK];
    }

    //load in threadblock reductions
    if (threadIdx.x < RADIX)
    {
        s_localHistogram[threadIdx.x] = globalHist[threadIdx.x + (radixShift << 5)] +
            passHist[threadIdx.x * gridDim.x + blockIdx.x] - s_warpHistograms[threadIdx.x];
    }
    __syncthreads();

    //scatter keys into shared memory
    #pragma unroll
    for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
        s_warpHistograms[offsets[i]] = keys[i];
    __syncthreads();

    //scatter runs of keys into device memory
    uint8_t digits[BIN_KEYS_PER_THREAD];
    if (blockIdx.x < gridDim.x - 1)
    {
        //store the digit of key in register
        #pragma unroll
        for (uint32_t i = 0, t = threadIdx.x; i < BIN_KEYS_PER_THREAD;
            ++i, t += blockDim.x)
        {
            uint32_t sortableBits = toBits(s_warpHistograms[t], radixShift);
            digits[i] = sortableBits >> radixShift & RADIX_MASK;
            alt[s_localHistogram[digits[i]] + t] = s_warpHistograms[t];
        }
        __syncthreads();

        //Load payloads into registers
        #pragma unroll
        for (uint32_t i = 0, t = getLaneId() + BIN_SUB_PART_START + BIN_PART_START;
            i < BIN_KEYS_PER_THREAD;
            ++i, t += LANE_COUNT)
        {
            keys[i] = sortPayload[t];
        }

        //scatter payloads into shared memory
        #pragma unroll
        for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
            s_warpHistograms[offsets[i]] = keys[i];
        __syncthreads();

        //Scatter the payloads into device
        #pragma unroll
        for (uint32_t i = 0, t = threadIdx.x; i < BIN_KEYS_PER_THREAD;
            ++i, t += blockDim.x)
        {
            altPayload[s_localHistogram[digits[i]] + t] = s_warpHistograms[t];
        }
    }

    // scatter with size check
    if (blockIdx.x == gridDim.x - 1)
    {
        const uint32_t finalPartSize = size - BIN_PART_START;
        //store the digit of key in register
        #pragma unroll
        for (uint32_t i = 0, t = threadIdx.x; i < BIN_KEYS_PER_THREAD;
            ++i, t += blockDim.x)
        {
            if (t < finalPartSize)
            {
                uint32_t sortableBits = toBits(s_warpHistograms[t], radixShift);
                digits[i] = sortableBits >> radixShift & RADIX_MASK;
                alt[s_localHistogram[digits[i]] + t] = s_warpHistograms[t];
            }
        }
        __syncthreads();

        //Load payloads into registers
        #pragma unroll
        for (uint32_t i = 0, t = getLaneId() + BIN_SUB_PART_START + BIN_PART_START;
            i < BIN_KEYS_PER_THREAD;
            ++i, t += LANE_COUNT)
        {
            if(t < size)
                keys[i] = sortPayload[t];
        }

        //scatter payloads into shared memory
        #pragma unroll
        for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
            s_warpHistograms[offsets[i]] = keys[i];
        __syncthreads();

        //Scatter the payloads into device
        #pragma unroll
        for (uint32_t i = 0, t = threadIdx.x; i < BIN_KEYS_PER_THREAD;
            ++i, t += blockDim.x)
        {
            if(t < finalPartSize)
                altPayload[s_localHistogram[digits[i]] + t] = s_warpHistograms[t];
        }
    }
}


static inline uint32_t divRoundUp(uint32_t x, uint32_t y) {
    return (x + y - 1) / y;
}


template <typename T>
void radix() {
    srand(time(NULL));

    const uint32_t k_maxSize = 7680;
    const uint32_t k_radix = RADIX;
    const uint32_t k_radixPasses = sizeof(uint32_t);
    const uint32_t k_partitionSize = PART_SIZE;
    const uint32_t k_upsweepThreads = 128;
    const uint32_t k_scanThreads = 128;
    const uint32_t k_downsweepThreads = 512;

    T* m_sort;
    uint32_t* m_sortPayload;
    T* m_alt;
    uint32_t* m_altPayload;
    uint32_t* m_globalHistogram;
    uint32_t* m_passHistogram;

    const uint32_t threadblocks = divRoundUp(k_maxSize, k_partitionSize);

    uint32_t sortsize = k_maxSize * sizeof(T);
    uint32_t payloadsize = k_maxSize * sizeof(uint32_t);

    // Allocate memories
    cudaMalloc(&m_sort, sortsize); // Input array
    cudaMalloc(&m_alt, sortsize); // alternate buffer for sorted output
    cudaMalloc(&m_sortPayload, payloadsize); // the sort payload
    cudaMalloc(&m_altPayload, payloadsize); // alternate buffer for sorted payload
    cudaMalloc(&m_globalHistogram, k_radix * k_radixPasses * sizeof(uint32_t)); // Global histogram
    cudaMalloc(&m_passHistogram, threadblocks * k_radix * sizeof(uint32_t)); // Local histogram - scanned offset from current pass?

    
    // Create some data
    T *msort_H = (T*)malloc(sortsize);
    uint32_t *mayload_H = (uint32_t*)malloc(payloadsize);
    for(int i = 0; i < k_maxSize; i++) {
        msort_H[i] = static_cast<T>(randnum(0, k_maxSize));
        mayload_H[i] = static_cast<uint32_t>(i);
    }

    printf("\nBEFORE[200] .........................\n");
    for(int i=0; i < 200; i++) {
        if(std::is_same<T, uint32_t>::value) {
            printf("[%u ", msort_H[i]);
        } else if(std::is_same<T, float>::value) {
            printf("[%f ", msort_H[i]);
        }
        printf("%u] ", mayload_H[i]);
    }
    printf("\n.....................................\n");

    cudaMemcpy(m_sort, msort_H, sortsize, cudaMemcpyHostToDevice);
    cudaMemcpy(m_sortPayload, mayload_H, payloadsize, cudaMemcpyHostToDevice);

    cudaMemset(m_globalHistogram, 0, k_radix * k_radixPasses * sizeof(uint32_t));
    cudaDeviceSynchronize();


    for(uint32_t shift=0; shift < k_radixPasses; shift++) {
        uint32_t pass = shift * 8;
        RadixUpsweep<T> <<<threadblocks, k_upsweepThreads>>> (m_sort, m_globalHistogram, m_passHistogram, k_maxSize, pass);
        RadixScan <<<k_radix, k_scanThreads>>> (m_passHistogram, threadblocks);
        RadixDownsweepPairs<T> <<<threadblocks, k_downsweepThreads>>> (m_sort, m_sortPayload, m_alt, m_altPayload,
            m_globalHistogram, m_passHistogram, k_maxSize, pass);

        std::swap(m_sort, m_alt);
        std::swap(m_sortPayload, m_altPayload);
    }

    cudaDeviceSynchronize();

    T *m_sort_h = (T*)malloc(sortsize);
    uint32_t *m_payl_h = (uint32_t*)malloc(payloadsize);

    cudaMemcpy(m_sort_h, m_sort, sortsize, cudaMemcpyDeviceToHost);
    cudaMemcpy(m_payl_h, m_sortPayload, payloadsize, cudaMemcpyDeviceToHost);

    // Validate
    printf("\nAFTER[200] .........................\n");
    T last = m_sort_h[0];
    uint32_t err = 0;

    for(int i=1; i < k_maxSize; ++i) {
        // if last one is greater than current one, or value at sorted index in original array != current value, its an error
        if(last <= m_sort_h[i] && msort_H[m_payl_h[i]] == m_sort_h[i]) {
            last = m_sort_h[i];
            continue;
        }
        err += 1;
        if(err < 10 && std::is_same<T, float>::value) {
            printf("Error[%d]: Index[%d] Last[%f] Current[%f] Val@Origin[%f] last.le(cur)[%d]\n", i, m_payl_h[i], last, m_sort_h[i], msort_H[m_payl_h[i]], last <= m_sort_h[i]);
        }
    }
    printf("\n.....................................\n");


    // Free memories
    free(msort_H);
    free(mayload_H);

    cudaFree(m_sort);
    cudaFree(m_alt);
    cudaFree(m_sortPayload);
    cudaFree(m_altPayload);
    cudaFree(m_globalHistogram);
    cudaFree(m_passHistogram);

    free(m_sort_h);
    free(m_payl_h);

    printf("%u errors", err);
}

int main() {
    // Test uint32_t
    radix<uint32_t>();
    // Test float
    radix<float>();
    return 0;
}