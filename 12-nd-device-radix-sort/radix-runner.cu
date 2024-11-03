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

template<typename T>
void swap(T* &a, T* &b){
    T* temp = a;
    a = b;
    b = temp;
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
            max = size <= 1024 ? 64 : (uint32_t)320000;
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
            h_sort[i] = random_range(min, max);
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
    // uint32_t numVecElemInBlock; // Vector elements per block
    uint32_t numThreadBlocks; // number of threadblocks to run for Upsweep and DownsweepPairs kernel

    uint32_t const numElemInBlock      = 64; // Elements per block
    uint32_t const numUpsweepThreads   = 32; // Num threads per upsweep kernel
    uint32_t const numScanThreads      = 16; // Num of threads for scan
    uint32_t const numDownsweepThreads = 64; // number of downsweep threads
    uint32_t const radix = RADIX;

    static Resources compute(uint32_t size, uint32_t type_size) {
        Resources res;

        // Query device properties
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        
        // Calculate shared memory needed for per-block histogram
        // This corresponds to __shared__ uint32_t s_globalHist[RADIX * 2] in the kernel
        // const uint32_t shared_hist_size = res.radix * 2 * sizeof(uint32_t);
        
        // // Calculate available shared memory for data processing
        // const uint32_t available_shared_mem = ((prop.sharedMemPerBlock - shared_hist_size) * 3) / 4;  // Use ~75% of remaining shared memory
        
        // Calculate part_size based on shared memory constraints
        // res.numElemInBlock = min((uint32_t)available_shared_mem / type_size, size);
        res.numThreadBlocks = (size + res.numElemInBlock - 1) / res.numElemInBlock;

        return res;
    }
};

template<typename T>
__global__ void printSize(T* d) {
    printf("Size: %lu", sizeof(d));
}

template<typename T>
uint32_t validate(const uint32_t size, bool dataseq = true) {
    printf("Validating for size[%u] and typeSize[%lu]\n", size, sizeof(T));
    uint32_t errors = 0;

    Resources res = Resources::compute(size, sizeof(uint32_t));
    printf("For size[%u] -------------\nnumThreadBlocks: %u numUpsweepThreads: %u numScanThreads: %u maxNumElementsInBlock: %u\n", size, res.numThreadBlocks, res.numUpsweepThreads, res.numScanThreads, res.numElemInBlock);
    
    cudaError_t c_ret;

    // Declarations
    T* d_sort;
    T* d_sortAlt;
    uint32_t* d_idx;
    uint32_t* d_idxAlt;
    
    uint32_t* d_globalHist;
    uint32_t* d_passHist;

    const uint32_t numPasses  = sizeof(T);
    const uint32_t sortSize   = size * sizeof(T);
    const uint32_t idxSize    = size * sizeof(uint32_t);
    const uint32_t radixSize  = RADIX * sizeof(uint32_t);

    T* h_sort       = (T*)malloc(sortSize);
    uint32_t* h_idx = (uint32_t*)malloc(idxSize);

    cudaMalloc(&d_sort, sortSize);
    cudaMalloc(&d_sortAlt, sortSize);
    cudaMalloc(&d_idx, idxSize);
    cudaMalloc(&d_idxAlt, idxSize);
    cudaMalloc(&d_globalHist, radixSize * numPasses);
    cudaMalloc(&d_passHist, radixSize * res.numThreadBlocks);

    // Create some data
    createData<T>(size, d_sort, d_idx, h_sort, h_idx, dataseq);

    cudaMemset(d_globalHist, 0,  radixSize * numPasses);
    cudaDeviceSynchronize();

    c_ret = cudaGetLastError();
    if (c_ret) {
        errors += 1;
        printf("[Pre-pass] Cuda Error: %s\n", cudaGetErrorString(c_ret));
        return errors;
    }

    // for(uint32_t pass=0; pass < numPasses; ++pass) {
    for(uint32_t pass=0; pass < 1; ++pass) {
        // Run `RadixUpsweep` kernel and validate
        uint32_t shift = pass * 8;
        printf("Pass[%u/ %u] Shift[%u]\n", pass, numPasses, shift);

        {
            // Get current sort data
            T* sortData = (T*)malloc(sortSize);
            c_ret = cudaMemcpy(sortData, d_sort, sortSize, cudaMemcpyDeviceToHost);
            if (c_ret) {
                printf("[Before Upsweep -> cudaMemcpy] Cuda Error: %s\n", cudaGetErrorString(c_ret));
                printSize<<<1, 1>>>(d_sort);
                c_ret = cudaGetLastError();
                if(c_ret) {
                    printf("Fuck this shit: %s\n",cudaGetErrorString(c_ret));
                }
                errors += 1;
                break;
            }

            c_ret = cudaDeviceSynchronize();
            if (c_ret) {
                printf("[Before Upsweep -> cudaDevSync] Cuda Error: %s\n", cudaGetErrorString(c_ret));
                errors += 1;
                break;
            }

            printf("Sort data now: \n");
            for(int i=0; i<32; i++) {
                if(std::is_same<T, float>::value)
                    printf("[%d %f] ", i, sortData[i]);
                else if(std::is_same<T, uint32_t>::value)
                    printf("[%d %u] ", i, sortData[i]);
            }
            printf("\n");

            RadixUpsweep<T><<<res.numThreadBlocks, res.numUpsweepThreads>>>(d_sort, d_globalHist, d_passHist, size, shift, res.numElemInBlock);
            c_ret = cudaGetLastError();
            if (c_ret) {
                printf("[Upsweep] Cuda Error: %s\n", cudaGetErrorString(c_ret));
                errors += 1;
                break;
            }

            uint32_t *cpuGlobHist = (uint32_t*)malloc(radixSize);
            uint32_t *gpuGlobHist = (uint32_t*)malloc(radixSize);
            uint32_t *cpuPassHist = (uint32_t*)malloc(radixSize * res.numThreadBlocks);
            uint32_t *gpuPassHist = (uint32_t*)malloc(radixSize * res.numThreadBlocks);

            // Initialize all zeroes
            for(int i=0; i<RADIX; ++i) {
                cpuGlobHist[i] = 0;
            }
            for(int i=0; i<RADIX * res.numThreadBlocks; ++i) {
                cpuPassHist[i] = 0;
            }

            // Compute CPU histogram
            for (uint32_t i = 0; i < size; i++) {
                uint32_t bits = toBitsCpu<T>(sortData[i]);
                uint32_t digit = (bits >> shift) & RADIX_MASK;
                cpuGlobHist[digit]++;
            }
            // Convert to exclusive prefix sum
            uint32_t prev = 0;
            for (uint32_t i = 0; i < RADIX; i++) {
                uint32_t current = cpuGlobHist[i];
                cpuGlobHist[i] = prev;
                prev += current;
            }
            for(uint32_t i = 0; i<res.numThreadBlocks; ++i) {
                uint32_t blockstart = i * res.numElemInBlock;
                uint32_t blockend   = min(blockstart + res.numElemInBlock, size);

                for(uint32_t j=blockstart; j < blockend; ++j) {
                    uint32_t bits = toBitsCpu<T>(sortData[j]);
                    uint32_t trgt = i * RADIX + ((bits >> shift) & RADIX_MASK);
                    cpuPassHist[trgt] += 1;
                }
            }

            c_ret = cudaMemcpy(gpuGlobHist, d_globalHist + (RADIX * pass), radixSize, cudaMemcpyDeviceToHost);
            if (c_ret) {
                printf("[Upsweep -> cudaMemcpy -> GpuHist] Cuda Error: %s\n", cudaGetErrorString(c_ret));
                errors += 1;
                break;
            }

            c_ret = cudaMemcpy(gpuPassHist, d_passHist, radixSize * res.numThreadBlocks, cudaMemcpyDeviceToHost);
            if (c_ret) {
                printf("[Upsweep -> cudaMemcpy -> GpuPassHist] Cuda Error: %s\n", cudaGetErrorString(c_ret));
                errors += 1;
                break;
            }
            
            c_ret = cudaDeviceSynchronize();
            if (c_ret) {
                printf("[Upsweep -> cudaDevSync -> GpuPassHist] Cuda Error: %s\n", cudaGetErrorString(c_ret));
                errors += 1;
                break;
            }
            
            printf("Validating `passHist`\n");
            bool passHistError = false;
            for(uint32_t block=0; block < res.numThreadBlocks; ++block) {
                uint32_t offset = block * RADIX;

                for(uint32_t digit=0; digit < RADIX; digit++) {
                    uint32_t v_idx = offset + digit;
                    if(cpuPassHist[v_idx] != gpuPassHist[v_idx]) {
                        errors++;
                        passHistError = true;
                        printf("Error @ Block[%u] Digit[%u]: Cpu[%u] Gpu[%u]\n", block, digit, cpuPassHist[v_idx], gpuPassHist[v_idx]);
                    }
                }
            }
            printf("Pass hist validation: %s\n", passHistError ? "FAIL" : "PASS");

            printf("Validating `globalHist`\n");
            bool globalHistError = false;
            for(uint32_t i=0; i<RADIX; ++i) {
                if(cpuGlobHist[i] != gpuGlobHist[i]) {
                    errors++;
                    globalHistError = true;
                    printf("Error @ Digit[%u]: Cpu[%u] Gpu[%u]\n", i, cpuGlobHist[i], gpuGlobHist[i]);
                }
            }
            printf("Global hist validation: %s\n", globalHistError ? "FAIL" : "PASS");
            
            free(cpuGlobHist);
            free(gpuGlobHist);
            free(cpuPassHist);
            free(gpuPassHist);
            free(sortData);
        }

        // Launch RadixScan kernel
        {
            uint32_t pass_hist_size = radixSize * res.numThreadBlocks;

            // Copy old state
            uint32_t* passHistBefore = (uint32_t*)malloc(pass_hist_size);
            c_ret = cudaMemcpy(passHistBefore, d_passHist, pass_hist_size, cudaMemcpyDeviceToHost);
            if (c_ret) {
                printf("[Before Scan -> cudaMemcpy -> passHistBefore] Cuda Error: %s\n", cudaGetErrorString(c_ret));
                errors += 1;
                break;
            }
            c_ret = cudaDeviceSynchronize();
            if (c_ret) {
                printf("[Before Scan -> cudaDevSync -> passHistBefore] Cuda Error: %s\n", cudaGetErrorString(c_ret));
                errors += 1;
                break;
            }

            RadixScan<<<RADIX, res.numScanThreads>>>(d_passHist, res.numThreadBlocks);
            c_ret = cudaGetLastError();
            if (c_ret) {
                printf("[Scan] Cuda Error: %s", cudaGetErrorString(c_ret));
                errors += 1;
                break;
            }

            printf("\nPeeking 16 digits in passHist: \n");
            for(uint32_t blk=0; blk<res.numThreadBlocks; ++blk) {
                uint32_t offset = RADIX * blk;
                printf("Block %u:\n", blk);
                for(uint32_t d=0; d<16; ++d) {
                    printf("[%u %u] ", d, passHistBefore[offset + d]);
                }
                printf("\n");
            }
            printf("\n");
        //     // Copy new state
        //     uint32_t* passHistGpu = (uint32_t*)malloc(pass_hist_size);
        //     uint32_t* passHistCpu = (uint32_t*)malloc(pass_hist_size);

        //     c_ret = cudaMemcpy(passHistGpu, d_passHist, pass_hist_size, cudaMemcpyDeviceToHost);
        //     if (c_ret) {
        //         printf("[Scan -> memCpy -> d_passHist] Cuda Error: %s", cudaGetErrorString(c_ret));
        //         errors += 1;
        //         break;
        //     }
        //     c_ret = cudaDeviceSynchronize();
        //     if (c_ret) {
        //         printf("[Scan -> cudaDevSync -> d_passHist] Cuda Error: %s", cudaGetErrorString(c_ret));
        //         errors += 1;
        //         break;
        //     }

        //     // Create cpu alternate values
        //     // Process each partition separately
        //     for (uint32_t r = 0; r < RADIX; r++) {
        //         uint32_t offset = r * res.numThreadBlocks;
                
        //         // First element remains the same
        //         passHistCpu[offset] = passHistBefore[offset];
                
        //         // Simple inclusive scan for this partition
        //         for (uint32_t i = 1; i < res.numThreadBlocks; i++) {
        //             passHistCpu[offset + i] = passHistCpu[offset + i - 1] + 
        //                                     passHistBefore[offset + i];
        //         }
        //     }

        //     for (uint32_t r = 0; r < RADIX; r++) {
        //         uint32_t offset = r * res.numThreadBlocks;
        //         if(r < 16) {
        //             printf("Hist@[%u]: %u\n", r, passHistGpu[r]);
        //         }
        //         for (uint32_t i = 0; i < res.numThreadBlocks; i++) {
        //             if (passHistGpu[offset + i] != passHistCpu[offset + i]) {
        //                 errors += 1;
        //                 if(errors < 10)
        //                     printf("Mismatch at partition %u, index %u: GPU = %u, CPU = %u\n", r, i, passHistGpu[offset + i], passHistCpu[offset + i]);
        //             }
        //         }
        //     }

        //     free(passHistBefore);
        //     free(passHistGpu);
        //     free(passHistCpu);
        }

        // Finally launch the Downsweep kernel
        // {
        //     // RadixDownsweep<<<res.numThreadBlocks, 256>>>(d_sort, d_sortAlt, d_idx, d_idxAlt, d_globalHist, d_passHist, size, shift);
        //     RadixDownsweep<<<res.numThreadBlocks, 256>>>(d_sort, d_sortAlt, d_globalHist, d_passHist, size, shift);
        //     c_ret = cudaGetLastError();
        //     if (c_ret) {
        //         printf("[Downsweep] Cuda Error: %s", cudaGetErrorString(c_ret));
        //         errors += 1;
        //         break;
        //     }

        //     c_ret = cudaDeviceSynchronize();
        //     if (c_ret) {
        //         printf("[Downsweep - cudaDivSync] Cuda Error: %s", cudaGetErrorString(c_ret));
        //         fprintf(stderr, "Error Code: %d\n", static_cast<int>(c_ret));
        //         fprintf(stderr, "Error String: %s\n", cudaGetErrorString(c_ret));
        //         errors += 1;
        //         break;
        //     }
        // }

        // // Swap after each pass
        // // swap(d_sort, d_sortAlt);
        // std::swap(d_sort, d_sortAlt);
        // std::swap(d_idx, d_idxAlt);

        if(errors > 0) {
            printf("Breaking at pass %u\n", pass);
            break;
        }
    }

    cudaFree(d_sort);
    cudaFree(d_sortAlt);
    cudaFree(d_idx);
    cudaFree(d_idxAlt);
    cudaFree(d_globalHist);
    cudaFree(d_passHist);

    free(h_sort);
    free(h_idx);
    return errors;
}

int main() {
    uint32_t sizes[] = { 67, 1024, 2048, 4096, 4113, 7680, 8192, 9216, 16000, 32000, 64000, 128000, 280000 };
    
    // First, test for UpsweepKernel is good?
    for(uint32_t i = 0; i < 1; i++) {
        // {
        //     printf("`uint32_t`: Upsweep Validation (sequential)\n");
        //     uint32_t errors = validate<uint32_t>(sizes[i]);
        //     if(errors > 0){
        //         printf("Errors: %u while validating upsweep for size[uint32_t][%u]\n", errors, sizes[i]);
        //         break;
        //     }
        // }

        {
            printf("`uint32_t`: Upsweep Validation (random)\n");
            uint32_t errors = validate<uint32_t>(sizes[i], false);
            if(errors > 0)
                printf("Errors: %u while validating upsweep for size[uint32_t][%u]\n", errors, sizes[i]);
        }

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