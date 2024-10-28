#include <stdint.h>
#include <stdio.h>
#include <tuple>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "radix.cu"

// Random float helper
static inline float random_float() {
    return (float)rand() / (float)RAND_MAX;
}

// Template declarations for different types
template<typename T>
inline T random_range(T min, T max);

// Specialization for uint8_t
template<>
inline uint8_t random_range(uint8_t min, uint8_t max) {
    return min + (rand() % (max - min + 1));
}

// Specialization for uint32_t
template<>
inline uint32_t random_range(uint32_t min, uint32_t max) {
    uint32_t range = max - min + 1;
    if (range > RAND_MAX) {
        uint32_t r = ((uint32_t)rand() << 16) | (uint32_t)rand();
        return min + (r % range);
    }
    return min + (rand() % range);
}

// Specialization for float
template<>
inline float random_range(float min, float max) {
    return min + random_float() * (max - min);
}

#ifdef HALF_FLOAT_SUPPORT
// Specialization for half float if needed
template<>
static inline __fp16 random_range(__fp16 min, __fp16 max) {
    float min_f = __half2float(min);
    float max_f = __half2float(max);
    return __float2(min_f + random_float() * (max_f - min_f));
}
#endif

#ifdef BRAIN_FLOAT_SUPPORT
// Specialization for half float if needed
template<>
static inline __nv_bfloat16 random_range(__nv_bfloat16 min, __nv_bfloat16 max) {
    float min_f = __bfloat162float(min);
    float max_f = __bfloat162float(max);
    return __float2bfloat16(min_f + random_float() * (max_f - min_f));
}
#endif

template<typename T>
void createData(uint32_t size, T* d_sort, uint32_t* d_idx, T* h_sort, uint32_t* h_idx, bool seq) {
    uint32_t sortsize = size * sizeof(T);
    uint32_t idxsize  = size * sizeof(uint32_t);

    T min;
    T max;

    if(!seq) {
        if(std::is_same<T, uint8_t>::value) {
            min = (uint8_t)0;
            max = (uint8_t)255;
        } else if(std::is_same<T, uint32_t>::value) {
            min = (uint32_t)0;
            max = (uint32_t)320000;
        } else if(std::is_same<T, float>::value) {
            min = (float)-512.0;
            max = (float)24000.0;
        }
        //  else if(std::is_same<T, __fp16>::value) {
        //     float mn = -32.0;
        //     float mx = 128.0f;
        //     min = __float2half(mn);
        //     max = __float2half(mx);
        // } else if(std::is_same<T, __nv_bfloat16>::value) {
        //     float mn = -64.0;
        //     float mx = 64.0f;
        //     min = __float2bfloat16(mn);
        //     max = __float2bfloat16(mx);
        // } 
    }

    for(uint32_t i=0; i<size; i++) {
        if(seq) {
            h_sort[i] = static_cast<T>(i);
        } else {
            random_range(min, max);
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
    uint32_t const numScanThreads = 256; // Number of scan threads

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
    printf("For size[%u] -------------\nnumThreadBlocks: %u numUpsweepThreads: %u numScanThreads: %u maxNumElementsInBlock: %u maxNumVecElementsInBlock: %u\n", size, res.numThreadBlocks, res.numUpsweepThreads, res.numScanThreads, res.numElemInBlock, res.numVecElemInBlock);
    
    // Declarations
    T* d_sort;
    T* d_sortAlt;
    uint32_t* d_idx;
    
    uint32_t* d_globalHist;
    uint32_t* d_passHist;

    uint32_t numPasses  = sizeof(T);
    uint32_t sortSize   = size * sizeof(T);
    uint32_t idxSize    = size * sizeof(uint32_t);
    uint32_t radixSize  = RADIX * sizeof(uint32_t);
    uint32_t scanShared = res.numScanThreads * sizeof(uint32_t);

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
        // Run `RadixUpsweep` kernel and validate
        uint32_t shift = pass * 8;
        RadixUpsweep<T><<<res.numThreadBlocks, res.numUpsweepThreads>>>(d_sort, d_globalHist, d_passHist, size, shift, res.numElemInBlock, res.numVecElemInBlock);
        {
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

        // Launch RadixScan kernel
        
        {
            uint32_t pass_hist_size = RADIX * sizeof(uint32_t) * res.numThreadBlocks;

            // Copy old state
            uint32_t* passHistBefore = (uint32_t*)malloc(pass_hist_size);
            cudaMemcpy(passHistBefore, d_passHist, pass_hist_size, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            
            RadixScan<<<RADIX, res.numScanThreads, scanShared>>>(d_passHist, res.numThreadBlocks);

            // Copy new state
            uint32_t* passHistGpu = (uint32_t*)malloc(pass_hist_size);
            uint32_t* passHistCpu = (uint32_t*)malloc(pass_hist_size);

            cudaMemcpy(passHistGpu, d_passHist, pass_hist_size, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            // Create cpu alternate values
            // Process each partition separately
            for (uint32_t r = 0; r < RADIX; r++) {
                uint32_t offset = r * res.numThreadBlocks;
                
                // First element remains the same
                passHistCpu[offset] = passHistBefore[offset];
                
                // Simple inclusive scan for this partition
                for (uint32_t i = 1; i < res.numThreadBlocks; i++) {
                    passHistCpu[offset + i] = passHistCpu[offset + i - 1] + 
                                            passHistBefore[offset + i];
                }
            }

            for (uint32_t r = 0; r < RADIX; r++) {
                uint32_t offset = r * res.numThreadBlocks;
                
                for (uint32_t i = 0; i < res.numThreadBlocks; i++) {
                    if (passHistGpu[offset + i] != passHistCpu[offset + i]) {
                        errors += 1;
                        printf("Mismatch at partition %u, index %u: GPU = %u, CPU = %u\n", r, i, passHistGpu[offset + i], passHistCpu[offset + i]);
                    }
                }
            }

            free(passHistBefore);
            free(passHistGpu);
            free(passHistCpu);
        }
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
    uint32_t sizes[] = { 16, 1024, 2048, 4096, 4113, 7680, 8192, 9216, 16000, 32000, 64000, 128000 };
    
    // First, test for UpsweepKernel is good?
    for(uint32_t i = 0; i < 8; i++) {
        {
            printf("`uint32_t`: Upsweep Validation (sequential)\n");
            uint32_t errors = validateUpsweep<uint32_t>(sizes[i]);
            if(errors > 0){
                printf("Errors: %u while validating upsweep for size[uint32_t][%u]\n", errors, sizes[i]);
                break;
            }
        }

        // {
        //     printf("`uint32_t`: Upsweep Validation (random)\n");
        //     uint32_t errors = validateUpsweep<uint32_t>(sizes[i], false);
        //     if(errors > 0)
        //         printf("Errors: %u while validating upsweep for size[uint32_t][%u]\n", errors, sizes[i]);
        // }

        // {
        //     printf("`float`: Upsweep Validation (sequential)\n");
        //     uint32_t errors = validateUpsweep<float>(sizes[i]);
        //     if(errors > 0)
        //         printf("Errors: %u while validating upsweep for size[float][%u]\n", errors, sizes[i]);
        // }

        // {
        //     printf("`float`: Upsweep Validation (random)\n");
        //     uint32_t errors = validateUpsweep<float>(sizes[i], false);
        //     if(errors > 0)
        //         printf("Errors: %u while validating upsweep for size[float][%u]\n", errors, sizes[i]);
        // }

        // {
        //     printf("`float16`: Upsweep Validation (seequential)\n");
        //     uint32_t errors = validateUpsweep<__fp16>(sizes[i], false);
        //     if(errors > 0)
        //         printf("Errors: %u while validating upsweep for size[fp16][%u]\n", errors, sizes[i]);
        // }

        // {
        //     printf("`float16`: Upsweep Validation (seequential)\n");
        //     uint32_t errors = validateUpsweep<__nv_bfloat16>(sizes[i], false);
        //     if(errors > 0)
        //         printf("Errors: %u while validating upsweep for size[bfloat16][%u]\n", errors, sizes[i]);
        // }
        // {
        //     uint32_t errors = validateUpsweep<half>(sizes[i]);
        //     if(errors > 0)
        //         printf("Errors: %u while validating upsweep for size[float16][%u]", errors, sizes[i]);
        // }
    }
    return 0;
}