#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <vector>

#include "radix.cu"

#define randnum(min, max) \
        ((rand() % (int)(((max) + 1) - (min))) + (min))

// #define WARP_INDEX          (threadIdx.x >> LANE_LOG)

// //For the upfront global histogram kernel
// #define PART_SIZE			7680
// #define VEC_PART_SIZE		(PART_SIZE / 4)

// //For the digit binning
// // #define BIN_PART_SIZE       PART_SIZE                               //Partition tile size in k_DigitBinning
// #define BIN_HISTS_SIZE      4096                                    //Total size of warp histograms in shared memory in k_DigitBinning
// // #define BIN_SUB_PART_SIZE   480                                     //Subpartition tile size of a single warp in k_DigitBinning
// #define BIN_KEYS_PER_THREAD 8                                       //Keys per thread in k_DigitBinning
// #define BIN_SUB_PART_START  (WARP_INDEX * BIN_SUB_PART_SIZE)        //Starting offset of a subpartition tile
// #define BIN_PART_START      (blockIdx.x * BIN_PART_SIZE)			//Starting offset of a partition tile



// __device__ __forceinline__ unsigned getLaneMaskLt() 
// {
//     unsigned mask;
//     asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
//     return mask;
// }



// //Warp scans
// __device__ __forceinline__ uint32_t InclusiveWarpScan(uint32_t val)
// {
//     #pragma unroll
//     for (int i = 1; i <= 16; i <<= 1) // 16 = LANE_COUNT >> 1
//     {
//         const uint32_t t = __shfl_up_sync(0xffffffff, val, i, 32);
//         if (getLaneId() >= i) val += t;
//     }

//     return val;
// }

// __device__ __forceinline__ uint32_t ActiveInclusiveWarpScan(uint32_t val)
// {
//     const uint32_t mask = __activemask();
//     #pragma unroll
//     for (int i = 1; i <= 16; i <<= 1)
//     {
//         const uint32_t t = __shfl_up_sync(mask, val, i, 32);
//         if (getLaneId() >= i) val += t;
//     }

//     return val;
// }



// template<typename T>
// __device__ inline T fromBits(uint32_t bits, uint32_t radixShift = 0) {
//     if constexpr (std::is_same<T, float>::value) {
//         uint32_t mask = ((bits >> 31) - 1) | 0x80000000;
//         return __uint_as_float(bits ^ mask);
//     }
//     else if constexpr (std::is_same<T, __half>::value) {
//         uint16_t shortBits = static_cast<uint16_t>(bits);
//         uint16_t mask = ((shortBits >> 15) - 1) | 0x8000;
//         return __ushort_as_half(shortBits ^ mask);
//     }
//     else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
//         uint16_t shortBits = static_cast<uint16_t>(bits);
//         uint16_t mask = ((shortBits >> 15) - 1) | 0x8000;
//         return __ushort_as_bfloat16(shortBits ^ mask);
//     }
//     else if constexpr (std::is_same<T, int64_t>::value) {
//         return static_cast<int64_t>(bits) << radixShift;
//     }
//     else {
//         return static_cast<T>(bits);
//     }
// }

// // Helper function to get type-specific maximum value
// template<typename T>
// __device__ inline T getTypeMax() {
//     if constexpr (std::is_same<T, float>::value) {
//         return INFINITY;
//     }
//     else if constexpr (std::is_same<T, __half>::value) {
//         return __float2half(INFINITY);
//     }
//     else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
//         return __float2bfloat16(INFINITY);
//     }
//     else if constexpr (std::is_same<T, int64_t>::value) {
//         return 0x7FFFFFFFFFFFFFFF;
//     } else if constexpr (std::is_same<T, unsigned char>::value) {
//         return 0xFF; // 255 in hex
//     } else if constexpr (std::is_same<T, u_int32_t>::value) {
//         return 0xFFFFFFFF;  // 4294967295 in hex
//     } else {
//         // This seems to be experimental
//         // calling a constexpr __host__ function("max") from a __device__ function("getTypeMax") is not allowed. The experimental flag '--expt-relaxed-constexpr' can be used to allow this.
        
//         // Shouldn't reach here
//         return static_cast<T>(-1);
//     }
// }

// template<typename T>
// __global__ void RadixUpsweep(
//     T* sort,
//     uint32_t* globalHist,
//     uint32_t* passHist,
//     uint32_t size,
//     uint32_t radixShift,
//     uint32_t partitionSize
// ) {
//     __shared__ uint32_t s_globalHist[RADIX * 2];

//     //clear shared memory
//     for (uint32_t i = threadIdx.x; i < RADIX * 2; i += blockDim.x)
//         s_globalHist[i] = 0;
//     __syncthreads();
    
//     //histogram
//     {
//         //64 threads : 1 histogram in shared memory
//         // Calculate this block's work range
//         uint32_t blockStart = blockIdx.x * partitionSize;
//         uint32_t blockEnd = min(blockStart + partitionSize, size);
//         uint32_t* s_wavesHist = &s_globalHist[threadIdx.x / 64 * RADIX];

//         for (uint32_t i = threadIdx.x + blockStart; i < blockEnd; i += blockDim.x) {
//             T val = sort[i];
//             uint32_t bits = toBits(val, radixShift);
//             atomicAdd(&s_wavesHist[bits >> radixShift & RADIX_MASK], 1);
//         }
//     //     if (blockIdx.x < gridDim.x - 1)
//     //     {
//     //         const uint32_t partEnd = (blockIdx.x + 1) * VEC_PART_SIZE;
            
//     //         // Vector load based on types
//     //         constexpr uint32_t typesize = sizeof(T);

//     //         // For uint32_t, float (maybe int32??)
//     //         if(typesize == 4) {
//     //             using VecT = typename std::conditional<std::is_same<T, float>::value, float4, uint4>::type;

//     //             for (uint32_t i = threadIdx.x + (blockIdx.x * VEC_PART_SIZE); i < partEnd; i += blockDim.x) {
//     //                 const VecT t = reinterpret_cast<VecT*>(sort)[i];
//     //                 // Convert to sortable bits based on type
//     //                 uint32_t x = toBits(t.x);
//     //                 uint32_t y = toBits(t.y);
//     //                 uint32_t z = toBits(t.z);
//     //                 uint32_t w = toBits(t.w);

//     //                 // if(i < 2) {
//     //                 //     if(std::is_same<T, float>::value) {
//     //                 //         // Debug prints
//     //                 //         uint32_t original_bits;
//     //                 //         memcpy(&original_bits, &t.x, sizeof(float));
//     //                 //         uint32_t after_transform = toBits(t.x);
//     //                 //         uint32_t after_shift = after_transform >> radixShift;
//     //                 //         uint32_t after_m = after_shift & RADIX_MASK;
//     //                 //         printf("State at [%u]: radixShift[%u] RADIX_MASK[%u]\nValue x: %f\nOriginal bits: 0x%08x\nAfter transform: 0x%08x\nAfter shift: 0x%08x\nAfter `after_shift >> RADIX_MASK`: %u\n", i, radixShift, RADIX_MASK, t.x, original_bits, after_transform, after_shift, after_m);
//     //                 //         // printf("\nFirst block[%u]: [%f %u\n%f %u\n%f %u\n%f %u]\n", i, t.x, x, t.y, y, t.z, z, t.w, w);
//     //                 //     } else
//     //                 //         printf("\nFirst block[%u]: [%u %u\n%u %u\n%u %u\n%u %u]\n", i, t.x, x, t.y, y, t.z, z, t.w, w);
//     //                 // }
//     //                 atomicAdd(&s_wavesHist[x >> radixShift & RADIX_MASK], 1);
//     //                 atomicAdd(&s_wavesHist[y >> radixShift & RADIX_MASK], 1);
//     //                 atomicAdd(&s_wavesHist[z >> radixShift & RADIX_MASK], 1);
//     //                 atomicAdd(&s_wavesHist[w >> radixShift & RADIX_MASK], 1);
//     //             }
//     //         }
//     //     }

//     //     if (blockIdx.x == gridDim.x - 1)
//     //     {
//     //         for (uint32_t i = threadIdx.x + (blockIdx.x * PART_SIZE); i < size; i += blockDim.x)
//     //         {
//     //             const T t = sort[i];
//     //             uint32_t bits = toBits(t, radixShift);
//     //             // if(i < 5) {
//     //             //     if(std::is_same<T, float>::value)
//     //             //         printf("Float Second block: [%u %f %u]\n", i, t, bits);
//     //             //     else
//     //             //         printf("Uint Second block: [%u %u %u]\n", i, t, bits);
//     //             // }
//     //             atomicAdd(&s_wavesHist[bits >> radixShift & RADIX_MASK], 1);
//     //         }
//     //     }
//     // }
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

// // __global__ void RadixUpsweep(
// //     uint32_t* sort,
// //     uint32_t* globalHist,
// //     uint32_t* passHist,
// //     uint32_t size,
// //     uint32_t radixShift
// // ) {
// //     __shared__ uint32_t s_globalHist[RADIX * 2];

// //     //clear shared memory
// //     for (uint32_t i = threadIdx.x; i < RADIX * 2; i += blockDim.x)
// //         s_globalHist[i] = 0;
// //     __syncthreads();
    
// //     //histogram
// //     {
// //         //64 threads : 1 histogram in shared memory
// //         uint32_t* s_wavesHist = &s_globalHist[threadIdx.x / 64 * RADIX];

// //         if (blockIdx.x < gridDim.x - 1)
// //         {
// //             const uint32_t partEnd = (blockIdx.x + 1) * VEC_PART_SIZE;
            
// //             for (uint32_t i = threadIdx.x + (blockIdx.x * VEC_PART_SIZE); i < partEnd; i += blockDim.x)
// //             {
// //                 const uint4 t = reinterpret_cast<uint4*>(sort)[i];
// //                 if(i < 5) {
// //                     printf("First block: [%u %u %u %u]\n", t.x, t.y, t.z, t.w);
// //                 }
// //                 atomicAdd(&s_wavesHist[t.x >> radixShift & RADIX_MASK], 1);
// //                 atomicAdd(&s_wavesHist[t.y >> radixShift & RADIX_MASK], 1);
// //                 atomicAdd(&s_wavesHist[t.z >> radixShift & RADIX_MASK], 1);
// //                 atomicAdd(&s_wavesHist[t.w >> radixShift & RADIX_MASK], 1);
// //             }
// //         }

// //         if (blockIdx.x == gridDim.x - 1)
// //         {
// //             for (uint32_t i = threadIdx.x + (blockIdx.x * PART_SIZE); i < size; i += blockDim.x)
// //             {
// //                 const uint32_t t = sort[i];
// //                 if(i < 5) {
// //                     printf("Second block: [%u %u]\n", i, t);
// //                 }
// //                 atomicAdd(&s_wavesHist[t >> radixShift & RADIX_MASK], 1);
// //             }
// //         }
// //     }
// //     __syncthreads();

// //     //reduce to the first hist, pass out, begin prefix sum
// //     for (uint32_t i = threadIdx.x; i < RADIX; i += blockDim.x)
// //     {
// //         s_globalHist[i] += s_globalHist[i + RADIX];
// //         passHist[i * gridDim.x + blockIdx.x] = s_globalHist[i];
// //         s_globalHist[i] = InclusiveWarpScanCircularShift(s_globalHist[i]);
// //     }	
// //     __syncthreads();

// //     if (threadIdx.x < (RADIX >> LANE_LOG))
// //         s_globalHist[threadIdx.x << LANE_LOG] = ActiveExclusiveWarpScan(s_globalHist[threadIdx.x << LANE_LOG]);
// //     __syncthreads();
    
// //     //Atomically add to device memory
// //     for (uint32_t i = threadIdx.x; i < RADIX; i += blockDim.x)
// //         atomicAdd(&globalHist[i + (radixShift << 5)], s_globalHist[i] + (getLaneId() ? __shfl_sync(0xfffffffe, s_globalHist[i - 1], 1) : 0));
// // }


// __global__ void RadixScan(
//     uint32_t* passHist,
//     uint32_t threadBlocks)
// {
//     __shared__ uint32_t s_scan[128];

//     uint32_t reduction = 0;
//     const uint32_t circularLaneShift = getLaneId() + 1 & LANE_MASK;
//     const uint32_t partitionsEnd = threadBlocks / blockDim.x * blockDim.x;
//     const uint32_t digitOffset = blockIdx.x * threadBlocks;

//     uint32_t i = threadIdx.x;
//     for (; i < partitionsEnd; i += blockDim.x)
//     {
//         s_scan[threadIdx.x] = passHist[i + digitOffset];
//         s_scan[threadIdx.x] = InclusiveWarpScan(s_scan[threadIdx.x]);
//         __syncthreads();

//         if (threadIdx.x < (blockDim.x >> LANE_LOG))
//         {
//             s_scan[(threadIdx.x + 1 << LANE_LOG) - 1] = 
//                 ActiveInclusiveWarpScan(s_scan[(threadIdx.x + 1 << LANE_LOG) - 1]);
//         }
//         __syncthreads();

//         passHist[circularLaneShift + (i & ~LANE_MASK) + digitOffset] =
//             (getLaneId() != LANE_MASK ? s_scan[threadIdx.x] : 0) +
//             (threadIdx.x >= LANE_COUNT ? __shfl_sync(0xffffffff, s_scan[threadIdx.x - 1], 0) : 0) +
//             reduction;

//         reduction += s_scan[blockDim.x - 1];
//         __syncthreads();
//     }

//     if(i < threadBlocks)
//         s_scan[threadIdx.x] = passHist[i + digitOffset];
//     s_scan[threadIdx.x] = InclusiveWarpScan(s_scan[threadIdx.x]);
//     __syncthreads();

//     if (threadIdx.x < (blockDim.x >> LANE_LOG))
//     {
//         s_scan[(threadIdx.x + 1 << LANE_LOG) - 1] =
//             ActiveInclusiveWarpScan(s_scan[(threadIdx.x + 1 << LANE_LOG) - 1]);
//     }
//     __syncthreads();

//     const uint32_t index = circularLaneShift + (i & ~LANE_MASK);
//     if (index < threadBlocks)
//     {
//         passHist[index + digitOffset] =
//             (getLaneId() != LANE_MASK ? s_scan[threadIdx.x] : 0) +
//             (threadIdx.x >= LANE_COUNT ?
//             s_scan[(threadIdx.x & ~LANE_MASK) - 1] : 0) +
//             reduction;
//     }
// }

// template<typename T>
// __global__ void RadixDownsweepPairs(
//     T* sort,
//     uint32_t* sortPayload,
//     T* alt, 
//     uint32_t* altPayload,
//     uint32_t* globalHist,
//     uint32_t* passHist,
//     uint32_t size, 
//     uint32_t radixShift)
// {
//     __shared__ uint32_t s_warpHistograms[BIN_PART_SIZE];
//     __shared__ uint32_t s_localHistogram[RADIX];
//     volatile uint32_t* s_warpHist = &s_warpHistograms[WARP_INDEX << RADIX_LOG];

//     //clear shared memory
//     for (uint32_t i = threadIdx.x; i < BIN_HISTS_SIZE; i += blockDim.x)
//         s_warpHistograms[i] = 0;

//     uint32_t baseIdx = getLaneId() + BIN_SUB_PART_START + BIN_PART_START;

//     //load keys
//     T keys[BIN_KEYS_PER_THREAD];
//     uint32_t sortableBits[BIN_KEYS_PER_THREAD];

//     if (blockIdx.x < gridDim.x - 1) {
//         #pragma unroll
//         for (uint32_t i = 0, t = baseIdx; i < BIN_KEYS_PER_THREAD; ++i, t += LANE_COUNT) {
//             keys[i] = t < size ? sort[t] : getTypeMax<T>();
//             sortableBits[i] = toBits(keys[i]);
//         }
//     } else if (blockIdx.x == gridDim.x - 1) {
//         //To handle input sizes not perfect multiples of the partition tile size,
//         //load "dummy" keys, which are keys with the highest possible digit
//         #pragma unroll
//         for (uint32_t i = 0, t = baseIdx; i < BIN_KEYS_PER_THREAD; ++i, t += LANE_COUNT) {
//             keys[i] = t < size ? sort[t] : getTypeMax<T>();
//             sortableBits[i] = toBits(keys[i]);
//         }
//     }

//     // WLMS (Work-efficient Local Memory Sort)
//     uint16_t offsets[BIN_KEYS_PER_THREAD];
//      #pragma unroll
//     for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i) {
//         unsigned warpFlags = 0xffffffff;
//         #pragma unroll
//         for (int k = 0; k < RADIX_LOG; ++k) {
//             const bool t2 = sortableBits[i] >> (k + radixShift) & 1;
//             warpFlags &= (t2 ? 0 : 0xffffffff) ^ __ballot_sync(0xffffffff, t2);
//         }
//         const uint32_t bits = __popc(warpFlags & getLaneMaskLt());
//         uint32_t preIncrementVal;
//         if (bits == 0)
//             preIncrementVal = atomicAdd((uint32_t*)&s_warpHist[(sortableBits[i] >> radixShift) & RADIX_MASK], __popc(warpFlags));

//         offsets[i] = __shfl_sync(0xffffffff, preIncrementVal, __ffs(warpFlags) - 1) + bits;
//     }
//     __syncthreads();

//     // Exclusive prefix sum up the warp histograms
//     if (threadIdx.x < RADIX) {
//         uint32_t reduction = s_warpHistograms[threadIdx.x];
//         for (uint32_t i = threadIdx.x + RADIX; i < BIN_HISTS_SIZE; i += RADIX) {
//             reduction += s_warpHistograms[i];
//             s_warpHistograms[i] = reduction - s_warpHistograms[i];
//         }
//         s_warpHistograms[threadIdx.x] = InclusiveWarpScanCircularShift(reduction);
//     }
//     __syncthreads();

//     if (threadIdx.x < (RADIX >> LANE_LOG))
//         s_warpHistograms[threadIdx.x << LANE_LOG] = ActiveExclusiveWarpScan(s_warpHistograms[threadIdx.x << LANE_LOG]);
//     __syncthreads();

//     if (threadIdx.x < RADIX && getLaneId())
//         s_warpHistograms[threadIdx.x] += __shfl_sync(0xfffffffe, s_warpHistograms[threadIdx.x - 1], 1);
//     __syncthreads();

//     //update offsets
//     // Update offsets
//     if (WARP_INDEX) {
//         #pragma unroll 
//         for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i) {
//             const uint32_t t2 = (sortableBits[i] >> radixShift) & RADIX_MASK;
//             offsets[i] += s_warpHist[t2] + s_warpHistograms[t2];
//         }
//     } else {
//         #pragma unroll
//         for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
//             offsets[i] += s_warpHistograms[(sortableBits[i] >> radixShift) & RADIX_MASK];
//     }

//     // Load threadblock reductions
//     if (threadIdx.x < RADIX) {
//         s_localHistogram[threadIdx.x] = globalHist[threadIdx.x + (radixShift << 5)] +
//             passHist[threadIdx.x * gridDim.x + blockIdx.x] - s_warpHistograms[threadIdx.x];
//     }
//     __syncthreads();

//     // Scatter keys into shared memory
//     #pragma unroll
//     for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
//         s_warpHistograms[offsets[i]] = sortableBits[i];
//     __syncthreads();

//     // Scatter runs of keys into device memory
//     uint8_t digits[BIN_KEYS_PER_THREAD];
//     const uint32_t finalPartSize = (blockIdx.x == gridDim.x - 1) ? size - BIN_PART_START : BIN_PART_SIZE;

//     // Store the digit of key in register and scatter keys
//     #pragma unroll
//     for (uint32_t i = 0, t = threadIdx.x; i < BIN_KEYS_PER_THREAD; ++i, t += blockDim.x) {
//         if (t < finalPartSize) {
//             uint32_t bits = s_warpHistograms[t];
//             digits[i] = (bits >> radixShift) & RADIX_MASK;
//             alt[s_localHistogram[digits[i]] + t] = fromBits<T>(bits);
//         }
//     }
//     __syncthreads();

//     // Load payloads into registers
//     #pragma unroll
//     for (uint32_t i = 0, t = baseIdx; i < BIN_KEYS_PER_THREAD; ++i, t += LANE_COUNT) {
//         if (t < size)
//             keys[i] = sortPayload[t];
//     }

//     // Scatter payloads into shared memory
//     #pragma unroll
//     for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
//         s_warpHistograms[offsets[i]] = keys[i];
//     __syncthreads();

//     // Scatter the payloads into device
//     #pragma unroll
//     for (uint32_t i = 0, t = threadIdx.x; i < BIN_KEYS_PER_THREAD; ++i, t += blockDim.x) {
//         if (t < finalPartSize)
//             altPayload[s_localHistogram[digits[i]] + t] = s_warpHistograms[t];
//     }
// }


static inline uint32_t divRoundUp(uint32_t x, uint32_t y) {
    return (x + y - 1) / y;
}

template<typename T>
uint32_t radix(
    uint32_t size,
    T* sort,
    uint32_t* sort_payload,
    bool with_validate = false
) {
    // Number of passes required for this sort
    const uint32_t radixPasses = sizeof(T);
    
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Calculate some static configs
    uint32_t upsweepThreads = min(256u, (uint32_t)prop.maxThreadsPerBlock);
    uint32_t scanThreads = min(256u, (uint32_t)prop.maxThreadsPerBlock);
    uint32_t downsweepThreads = min(512u, (uint32_t)prop.maxThreadsPerBlock);

    // Calculate optimal partition sizes
    uint32_t maxSharedMemPerBlock = prop.sharedMemPerBlock;
    // uint32_t numSMs = prop.multiProcessorCount;

    // Partition size should be multiple of (ITEMS_PER_THREAD * WARP_SIZE)
    // but not exceed shared memory constraints
    uint32_t partitionSize = ((maxSharedMemPerBlock / sizeof(uint32_t)) / RADIX) * 
                              (ITEMS_PER_THREAD * WARP_SIZE);
    partitionSize = max(2048u, partitionSize);
    // Sub-partition size for better memory coalescing
    // TODO: revisit this
    // uint32_t subPartitionSize = partitionSize / 
    //                              (downsweepThreads / WARP_SIZE);


    // Declarations
    T* d_sort;
    T* d_alt;
    uint32_t* d_payload;
    uint32_t* d_altPayload;
    uint32_t* d_globalHist;
    uint32_t* d_passHist;

    uint32_t sortSize = size * sizeof(T);
    uint32_t payloadSize = size * sizeof(uint32_t);

    uint32_t histSize = RADIX * radixPasses * sizeof(uint32_t);
    
    // Allocate device memory
    cudaMalloc(&d_sort, sortSize);
    cudaMalloc(&d_payload, payloadSize);
    cudaMalloc(&d_globalHist, histSize);
    cudaMalloc(&d_passHist, RADIX * sizeof(uint32_t) * divRoundUp(size, partitionSize));
    cudaMalloc(&d_alt, sortSize);
    cudaMalloc(&d_altPayload, payloadSize);

    // Num thread blocks
    uint32_t numBlocks = divRoundUp(size, partitionSize);

    // Copy data on to device memory
    cudaMemcpy(d_sort, sort, sortSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_payload, sort_payload, payloadSize, cudaMemcpyHostToDevice);

    cudaMemset(d_globalHist, 0, histSize);
    cudaDeviceSynchronize();

    for(uint32_t shift = 0; shift < radixPasses; shift++) {
        uint32_t pass = shift * RADIX_LOG;

        RadixUpsweep<T><<<numBlocks, upsweepThreads>>>(d_sort, d_globalHist, d_passHist, size, pass, partitionSize);
    }

    // Free Cuda memory
    cudaFree(d_sort);
    cudaFree(d_payload);
    cudaFree(d_alt);
    cudaFree(d_altPayload);
    cudaFree(d_globalHist);
    cudaFree(d_passHist);
    return 0;
}
// template <typename T>
// void radix(
//     uint32_t k_maxSize,
//     T* m_sort_h,
//     uint32_t* m_sortPayload_h
// ) {


//     for(uint32_t shift=0; shift < k_radixPasses; shift++) {

//         RadixScan <<<k_radix, scanThreads>>> (m_passHistogram, threadblocks);
//         RadixDownsweepPairs<T> <<<threadblocks, downsweepThreads>>> (m_sort, m_sortPayload, m_alt, m_altPayload,
//             m_globalHistogram, m_passHistogram, k_maxSize, pass);

//         std::swap(m_sort, m_alt);
//         std::swap(m_sortPayload, m_altPayload);
//     }

//     cudaDeviceSynchronize();

//     T *m_sort_res = (T*)malloc(sortsize);
//     uint32_t *m_sortPayload_res = (uint32_t*)malloc(payloadsize);

//     cudaMemcpy(m_sort_res, m_sort, sortsize, cudaMemcpyDeviceToHost);
//     cudaMemcpy(m_sortPayload_res, m_sortPayload, payloadsize, cudaMemcpyDeviceToHost);

//     // Validate
//     printf("\nAFTER .........................\n");
//     T last = m_sort_res[0];
//     uint32_t err = 0;
    
//     for(uint32_t i=1; i < k_maxSize; ++i) {
//         T current = m_sort_res[i];
//         uint32_t idx_sorted = m_sortPayload_res[i];
//         T original = static_cast<T>(200000);
//         if (idx_sorted < k_maxSize)
//             original = m_sort_h[idx_sorted];

//         // if last one is greater than current one, or value at sorted index in original array != current value, its an error
//         if(last > current || current != original) {
//             err += 1;
//             if(err < 10 && std::is_same<T, float>::value) {
//                 printf("Error@%u: Last[%f] Current[%f] OriginIdx[%u] Val@Origin[%f]\n", i, last, current, idx_sorted, original);
//                 printf("Last <= current %d\npaylodidx < k_maxSize: %d\norigin == current: %d\n", last <= current, idx_sorted < k_maxSize, current == original);
//             }
//         }
//         last = current;
//     }
//     printf("\n.....................................\n");


//     // Free memories

//     cudaFree(m_sort);
//     cudaFree(m_alt);
//     cudaFree(m_sortPayload);
//     cudaFree(m_altPayload);
//     cudaFree(m_globalHistogram);
//     cudaFree(m_passHistogram);

//     free(m_sort_res);
//     free(m_sortPayload_res);

//     printf("Size %u with %u errors", k_maxSize, err);
// }

void bit_32_data(uint32_t size, uint32_t* data, uint32_t* idxs) {
    for(uint32_t i = 0; i < size; i++) {
        data[i] = static_cast<uint32_t>(randnum(0, size));
        idxs[i] = static_cast<uint32_t>(i);
    }
}

// Helper functions for bit conversions
template<typename T>
inline uint32_t toBitsCpu(T val) {
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


template<typename T>
uint32_t validateRadixUpsweep(uint32_t size, uint32_t* data, uint32_t pass) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    T* sort = (T*)malloc(size * sizeof(T));
    for(uint32_t i=0; i < size; i++) {
        sort[i] = static_cast<T>(data[i]);
    }

    // Calculate some static configs
    uint32_t upsweepThreads = min(256u, (uint32_t)prop.maxThreadsPerBlock);
    // Calculate optimal partition sizes
    uint32_t maxSharedMemPerBlock = prop.sharedMemPerBlock;
    // uint32_t numSMs = prop.multiProcessorCount;

    // Partition size should be multiple of (ITEMS_PER_THREAD * WARP_SIZE)
    // but not exceed shared memory constraints
    uint32_t partitionSize = ((maxSharedMemPerBlock / sizeof(uint32_t)) / RADIX) * 
                              (ITEMS_PER_THREAD * WARP_SIZE);
    partitionSize = max(2048u, partitionSize);
    
    // Num thread blocks
    uint32_t numBlocks = divRoundUp(size, partitionSize);

    // Declarations
    T* d_sort;
    uint32_t* d_globalHist;
    uint32_t* d_passHist;

    uint32_t sortSize = size * sizeof(T);

    uint32_t histSize = RADIX * sizeof(uint32_t);
    
    // Allocate device memory
    cudaMalloc(&d_sort, sortSize);
    cudaMalloc(&d_globalHist, histSize * sizeof(T));
    cudaMalloc(&d_passHist, histSize * numBlocks);

    // Copy data on to device memory
    cudaMemcpy(d_sort, sort, sortSize, cudaMemcpyHostToDevice);

    // Clear all histogram memory
    cudaMemset(d_globalHist, 0, histSize);
    cudaMemset(d_passHist, 0, histSize * numBlocks);
    cudaDeviceSynchronize();

    RadixUpsweep<T><<<numBlocks, upsweepThreads>>>(d_sort, d_globalHist, d_passHist, size, pass, partitionSize);
    cudaDeviceSynchronize();
    
    // uint32_t* gpuHist = (uint32_t*)malloc(histSize);
    // cudaMemcpy(gpuHist, d_globalHist, histSize, cudaMemcpyDeviceToHost);
    uint32_t* cpuHist = (uint32_t*)malloc(RADIX * sizeof(uint32_t));
    uint32_t* gpuHist = (uint32_t*)malloc(RADIX * sizeof(uint32_t));

    cudaMemcpy(gpuHist, 
               d_globalHist + (pass * RADIX),  // Offset to current pass
               RADIX * sizeof(uint32_t), 
               cudaMemcpyDeviceToHost);
    
    // Compute CPU histogram
    for (uint32_t i = 0; i < size; i++) {
        uint32_t bits = toBitsCpu(sort[i]);
        uint32_t digit = (bits >> pass) & RADIX_MASK;
        cpuHist[digit]++;
    }

    uint32_t totalDiff = 0;
    bool match = true;
    // Verify results
    for(uint32_t i = 0; i < RADIX; i++) {
        if(gpuHist[i] != cpuHist[i]) {
            printf("Mismatch @ bin: %u -- CPU[%u] -- GPU[%u]\n", i, cpuHist[i], gpuHist[i]);
            totalDiff += std::abs((int)gpuHist[i] - (int)cpuHist[i]);
            match = false;
        }
    }

    if(match) {
        printf("All good for size[%u] and pass[%u]", size, pass);
    } else {
        printf("Error: size[%u] and pass[%u]: %u", size, pass, totalDiff);
    }

    // Free Cuda memory
    cudaFree(d_sort);
    cudaFree(d_globalHist);
    cudaFree(d_passHist);

    free(cpuHist);
    free(gpuHist);

    return totalDiff;
}

int main() {
    uint32_t sizes[] = { 2048, 3072, 7680, 15360 };
    
    for(int i=0; i<1; i++) {
        // First for 32 bits
        uint32_t* data = (uint32_t*)malloc(sizes[i] * sizeof(uint32_t));
        uint32_t* idxs = (uint32_t*)malloc(sizes[i] * sizeof(uint32_t));
        
        bit_32_data(sizes[i], data, idxs);
        for(uint32_t k=0; k<20;k++) {
            printf("%u ", data[k]);
        }
        printf("\nRunning validation");
        for(uint32_t j=0; j < 1; j++) {
            validateRadixUpsweep<uint32_t>(sizes[j], data, j * 8);
        }
    }

    // for(int i=0; i<N_TESTS; i++) {
    //     // First for 32 bits
    //     uint32_t* data = (uint32_t*)malloc(sizes[i] * sizeof(uint32_t));
    //     uint32_t* idxs = (uint32_t*)malloc(sizes[i] * sizeof(uint32_t));

    //     bit_32_data(sizes[i], data, idxs);
    //     printf("\nTesting for size[%u] uint32_t", sizes[i]);
    //     // Test uint32_t
    //     uint32_t errors = radix<uint32_t>(sizes[i], data, idxs, true);

    //     // Test float
    //     // cast uint32_t to float
    //     float* float_data = (float*)malloc(sizes[i] * sizeof(float));
    //     for(uint32_t j=0; j<sizes[i]; j++) {
    //         float_data[j] = static_cast<float>(data[j]);
    //     }
    //     free(data);
    //     printf("\nTesting for size[%u] float", sizes[i]);
    //     radix<float>(sizes[i], float_data, idxs, true);
    //     free(float_data);
    // }
    
    return 0;
}