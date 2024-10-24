#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

__global__ void RadixUpsweep(
    uint32_t* sort,
    uint32_t* globalHist,
    uint32_t* passHist,
    uint32_t size,
    uint32_t radixShift
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Debug Info:\n");
        printf("Grid Size: %d, Block Size: %d\n", gridDim.x, blockDim.x);
        printf("Input Size: %u, RadixShift: %u\n", size, radixShift);
        printf("RADIX: %d, VEC_PART_SIZE: %d\n", RADIX, VEC_PART_SIZE);
    }
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
            
            for (uint32_t i = threadIdx.x + (blockIdx.x * VEC_PART_SIZE); i < partEnd; i += blockDim.x)
            {
                const uint4 t = reinterpret_cast<uint4*>(sort)[i];
                // if (threadIdx.x == 0 && blockIdx.x == 0 && i == 0) {
                //     printf("First 4 elements:\n");
                //     printf("[%u %u %u %u]\n", t.x, t.y, t.z, t.w);
                //     printf("After shift & mask:\n");
                //     printf("[%u %u %u %u]\n", 
                //         t.x >> radixShift & RADIX_MASK,
                //         t.y >> radixShift & RADIX_MASK,
                //         t.z >> radixShift & RADIX_MASK,
                //         t.w >> radixShift & RADIX_MASK);
                // }
                atomicAdd(&s_wavesHist[t.x >> radixShift & RADIX_MASK], 1);
                atomicAdd(&s_wavesHist[t.y >> radixShift & RADIX_MASK], 1);
                atomicAdd(&s_wavesHist[t.z >> radixShift & RADIX_MASK], 1);
                atomicAdd(&s_wavesHist[t.w >> radixShift & RADIX_MASK], 1);
            }
        }

        if (blockIdx.x == gridDim.x - 1)
        {
            for (uint32_t i = threadIdx.x + (blockIdx.x * PART_SIZE); i < size; i += blockDim.x)
            {
                const uint32_t t = sort[i];
                atomicAdd(&s_wavesHist[t >> radixShift & RADIX_MASK], 1);
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

__global__ void RadixDownsweepPairs(
    uint32_t* sort,
    uint32_t* sortPayload,
    uint32_t* alt, 
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
    uint32_t keys[BIN_KEYS_PER_THREAD];
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
    if (blockIdx.x == gridDim.x - 1)
    {
        #pragma unroll
        for (uint32_t i = 0, t = getLaneId() + BIN_SUB_PART_START + BIN_PART_START; i < BIN_KEYS_PER_THREAD; ++i, t += LANE_COUNT)
            keys[i] = t < size ? sort[t] : 0xffffffff;
    }
    __syncthreads();

    //WLMS
    uint16_t offsets[BIN_KEYS_PER_THREAD];
    #pragma unroll
    for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
    {
        unsigned warpFlags = 0xffffffff;
        #pragma unroll
        for (int k = 0; k < RADIX_LOG; ++k)
        {
            const bool t2 = keys[i] >> k + radixShift & 1;
            warpFlags &= (t2 ? 0 : 0xffffffff) ^ __ballot_sync(0xffffffff, t2);
        }
        const uint32_t bits = __popc(warpFlags & getLaneMaskLt());
        uint32_t preIncrementVal;
        if (bits == 0)
            preIncrementVal = atomicAdd((uint32_t*)&s_warpHist[keys[i] >> radixShift & RADIX_MASK], __popc(warpFlags));

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
            const uint32_t t2 = keys[i] >> radixShift & RADIX_MASK;
            offsets[i] += s_warpHist[t2] + s_warpHistograms[t2];
        }
    }
    else
    {
        #pragma unroll
        for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
            offsets[i] += s_warpHistograms[keys[i] >> radixShift & RADIX_MASK];
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
            digits[i] = s_warpHistograms[t] >> radixShift & RADIX_MASK;
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
                digits[i] = s_warpHistograms[t] >> radixShift & RADIX_MASK;
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

void radix() {
    srand(time(NULL));

    const uint32_t k_maxSize = PART_SIZE;
    const uint32_t k_radix = RADIX;
    const uint32_t k_radixPasses = sizeof(uint32_t);
    const uint32_t k_partitionSize = PART_SIZE;
    const uint32_t k_upsweepThreads = 128;
    const uint32_t k_scanThreads = 128;
    const uint32_t k_downsweepThreads = 512;
    const uint32_t k_valPartSize = BIN_HISTS_SIZE;

    uint32_t* m_sort;
    uint32_t* m_sortPayload;
    uint32_t* m_alt;
    uint32_t* m_altPayload;
    uint32_t* m_globalHistogram;
    uint32_t* m_passHistogram;

    const uint32_t threadblocks = divRoundUp(k_maxSize, k_partitionSize);

    // Allocate memories
    cudaMalloc(&m_sort, k_maxSize * sizeof(uint32_t)); // Input array
    cudaMalloc(&m_alt, k_maxSize * sizeof(uint32_t)); // alternate buffer for sorted output
    cudaMalloc(&m_sortPayload, k_maxSize * sizeof(uint32_t)); // the sort payload
    cudaMalloc(&m_altPayload, k_maxSize * sizeof(uint32_t)); // alternate buffer for sorted payload
    cudaMalloc(&m_globalHistogram, k_radix * k_radixPasses * sizeof(uint32_t)); // Global histogram
    cudaMalloc(&m_passHistogram, threadblocks * k_radix * sizeof(uint32_t)); // Local histogram - scanned offset from current pass?
    
    // Create some data
    {
        uint32_t *msort_H = (uint32_t*)malloc(k_maxSize * sizeof(uint32_t));
        uint32_t *mayload_H = (uint32_t*)malloc(k_maxSize * sizeof(uint32_t));
        for(int i = 0; i < k_maxSize; i++) {
            msort_H[i] = static_cast<uint32_t>(randnum(0, k_maxSize));
            mayload_H[i] = static_cast<uint32_t>(i);
        }

        printf("\nBEFORE[200] .........................\n");
        for(int i=0; i < 200; i++) {
            printf("[%u %u] ", msort_H[i], mayload_H[i]);
        }
        printf("\n.....................................\n");

        cudaMemcpy(&m_sort, &msort_H, k_maxSize * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(&m_sortPayload, &mayload_H, k_maxSize * sizeof(uint32_t), cudaMemcpyHostToDevice);

        free(msort_H);
        free(mayload_H);
    }

    cudaMemset(m_globalHistogram, 0, k_radix * k_radixPasses * sizeof(uint32_t));
    cudaDeviceSynchronize();

    printf("\nNum Passes: %u ThreadBlocks: %u\n", k_radixPasses, threadblocks);
    for(uint32_t shift=0; shift < k_radixPasses; shift++) {
        uint32_t pass = shift * 8;
        printf("Pass: %u[%u]\n", shift, pass);
        RadixUpsweep <<<threadblocks, k_upsweepThreads>>> (m_sort, m_globalHistogram, m_passHistogram, k_maxSize, pass);
        RadixScan <<<k_radix, k_scanThreads>>> (m_passHistogram, threadblocks);
        RadixDownsweepPairs <<<threadblocks, k_downsweepThreads>>>(m_sort, m_sortPayload, m_alt, m_altPayload,
            m_globalHistogram, m_passHistogram, k_maxSize, pass);
    }

    cudaDeviceSynchronize();

    uint32_t *m_sort_h = (uint32_t*)malloc(k_maxSize * sizeof(uint32_t));
    uint32_t *m_payl_h = (uint32_t*)malloc(k_maxSize * sizeof(uint32_t));

    uint32_t *m_sort_h_alt = (uint32_t*)malloc(k_maxSize * sizeof(uint32_t));
    uint32_t *m_payl_h_alt = (uint32_t*)malloc(k_maxSize * sizeof(uint32_t));

    cudaMemcpy(&m_sort_h, &m_sort, k_maxSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&m_payl_h, &m_sortPayload, k_maxSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaMemcpy(&m_sort_h_alt, &m_alt, k_maxSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&m_payl_h_alt, &m_altPayload, k_maxSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("\nAFTER[200] .........................\n");
    for(int i=0; i < 200; i++) {
        printf("[%u %u | %u %u] ", m_sort_h[i], m_payl_h[i], m_sort_h_alt[i], m_payl_h_alt[i]);
    }
    printf("\n.....................................\n");


    // Free memories
    cudaFree(m_sort);
    cudaFree(m_alt);
    cudaFree(m_sortPayload);
    cudaFree(m_altPayload);
    cudaFree(m_globalHistogram);
    cudaFree(m_passHistogram);

    free(m_sort_h);
    free(m_payl_h);

    free(m_sort_h_alt);
    free(m_payl_h_alt);
}

int main() {
    radix();
    return 0;
}