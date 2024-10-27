#include <stdint.h>
#include <stdio.h>
#include <tuple>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "radix.cu"

template<typename T>
void createData(uint32_t size, T* d_sort, uint32_t* d_idx, T* h_sort, uint32_t* h_idx, bool seq) {
    uint32_t sortsize = size * sizeof(T);
    uint32_t idxsize  = size * sizeof(uint32_t);


    for(uint32_t i=0; i<size; i++) {
        if(seq) {
            h_sort[i] = static_cast<T>(i);
        } else {
            // TODO
        }
        h_idx[i] = i;
    }

    cudaMemcpy(d_sort, h_sort, sortsize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx, h_idx, idxsize, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
}

// Helper functions for bit conversions
template<typename T>
inline uint32_t toBitsCpu(T val) {
    if constexpr (std::is_same<T, float>::value) {
        uint32_t fuint;
        memcpy(&fuint, &val, sizeof(float));
        return fuint ^ ((fuint >> 31) | 0x80000000);
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

// Calculate resources to run
struct Resources {
    uint32_t numElemInBlock; // Elements per block
    uint32_t numVecElemInBlock; // Vector elements per block
    uint32_t numThreadBlocks; // number of threadblocks to run for Upsweep and DownsweepPairs kernel
    uint32_t const numUpsweepThreads = 256; // Num threads per upsweep kernel

    uint32_t const radix = RADIX;

    static Resources compute(uint32_t size, uint32_t type_size) {
        Resources res;

        // Query device properties
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        
        // Calculate shared memory needed for per-block histogram
        // This corresponds to __shared__ uint32_t s_globalHist[RADIX * 2] in the kernel
        const uint32_t shared_hist_size = res.radix * 2 * sizeof(uint32_t);

        // Calculate available shared memory for data processing
        const uint32_t available_shared_mem = (prop.sharedMemPerBlock - shared_hist_size) * 3 / 4;  // Use ~75% of remaining shared memory

        // Calculate part_size based on shared memory constraints
        res.numElemInBlock = available_shared_mem / type_size;
        // For 4-byte types, adjust vec_part_size for vector loads
        res.numVecElemInBlock = (type_size == 4) ? 
            res.numElemInBlock / 4 : res.numElemInBlock;

        res.numThreadBlocks = (size + res.numElemInBlock - 1) / res.numElemInBlock;
        return res;
    }
};

template<typename T>
uint32_t validateUpsweep(uint32_t size, bool dataseq = true) {
    printf("Validating upsweep for size[%u] and typeSize[%lu]\n", size, sizeof(T));
    uint32_t errors = 0;

    Resources res = Resources::compute(size, sizeof(uint32_t));
    printf("For size[%u] -------------\nnumThreadBlocks: %u numUpsweepThreads: %u numElementsInBlock: %u numVecElementsInBlock: %u\n", size, res.numThreadBlocks, res.numUpsweepThreads, res.numElemInBlock, res.numVecElemInBlock);
    
    // Declarations
    T* d_sort;
    T* d_sortAlt;
    uint32_t* d_idx;
    
    uint32_t* d_globalHist;
    uint32_t* d_passHist;

    uint32_t numPasses = sizeof(T);
    uint32_t sortSize  = size * sizeof(T);
    uint32_t idxSize   = size * sizeof(uint32_t);
    uint32_t radixSize = RADIX * sizeof(uint32_t);

    T* h_sort       = (T*)malloc(sortSize);
    uint32_t* h_idx = (uint32_t*)malloc(idxSize);

    cudaMalloc(&d_sort, sortSize);
    cudaMalloc(&d_sortAlt, sortSize);
    cudaMalloc(&d_idx, idxSize);
    // cudaMalloc(&d_sortAlt, sortSize);
    cudaMalloc(&d_globalHist, radixSize * numPasses);
    cudaMalloc(&d_passHist, radixSize * res.numThreadBlocks);

    // Create some data
    createData<T>(size, d_sort, d_idx, h_sort, h_idx, dataseq);

    cudaMemset(d_globalHist, 0,  radixSize * numPasses);
    cudaDeviceSynchronize();

    for(uint32_t pass=0; pass < numPasses; pass++) {
        uint32_t shift = pass * 8;
        RadixUpsweep<T><<<res.numThreadBlocks, res.numUpsweepThreads>>>(d_sort, d_globalHist, d_passHist, size, shift, res.numElemInBlock, res.numVecElemInBlock);

        uint32_t *cpuHist = (uint32_t*)malloc(radixSize);
        uint32_t *gpuHist = (uint32_t*)malloc(radixSize);

        for(int i=0; i<RADIX; i++) {
            cpuHist[i] = 0;
        }

        // Compute CPU histogram
        for (uint32_t i = 0; i < size; i++) {
            uint32_t bits = toBitsCpu<T>(h_sort[i]);
            uint32_t digit = (bits >> shift) & RADIX_MASK;
            cpuHist[digit]++;
        }
        // Convert to exclusive prefix sum
        uint32_t prev = 0;
        for (uint32_t i = 0; i < RADIX; i++) {
            uint32_t current = cpuHist[i];
            cpuHist[i] = prev;
            prev += current;
        }

        cudaMemcpy(gpuHist, d_globalHist + (RADIX * pass), radixSize, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        
        for(uint32_t i=0; i<RADIX; i++) {
            if(cpuHist[i] != gpuHist[i]) {
                errors += 1;
                printf("Error[bin %u/ radixShift %u]: CPU[%u] GPU[%u]\n", i, shift, cpuHist[i], gpuHist[i]);
            }
        }
        
        free(cpuHist);
        free(gpuHist);
    }

    cudaFree(d_sort);
    cudaFree(d_sortAlt);
    cudaFree(d_idx);
    cudaFree(d_globalHist);
    cudaFree(d_passHist);

    free(h_sort);
    free(h_idx);
    return errors;
}

int main() {
    uint32_t sizes[] = { 16, 1024, 2048, 4096, 4113, 7680, 8192, 32000, 64000, 128000 };
    
    // First, test for UpsweepKernel is good?
    for(uint32_t i = 0; i < 10; i++) {
        {
            printf("`uint32_t`: Upsweep Validation\n");
            uint32_t errors = validateUpsweep<uint32_t>(sizes[i]);
            if(errors > 0)
                printf("Errors: %u while validating upsweep for size[uint32_t][%u]\n", errors, sizes[i]);
        }

        {
            printf("`float`: Upsweep Validation\n");
            uint32_t errors = validateUpsweep<float>(sizes[i]);
            if(errors > 0)
                printf("Errors: %u while validating upsweep for size[float][%u]\n", errors, sizes[i]);
        }

        // {
        //     uint32_t errors = validateUpsweep<half>(sizes[i]);
        //     if(errors > 0)
        //         printf("Errors: %u while validating upsweep for size[float16][%u]", errors, sizes[i]);
        // }
    }
    return 0;
}