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
bool createData(uint32_t size, T* d_sort, uint32_t* d_idx, T* h_sort, uint32_t* h_idx, bool seq, bool withId) {
    uint32_t sortsize = size * sizeof(T);
    uint32_t idxsize  = withId ? size * sizeof(uint32_t) : 0;

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
            min = -24000.0f;
            max = 24000.0f;
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
        if (withId)
            h_idx[i] = i;
    }

    cudaError_t cerr;
    
    cerr = cudaMemcpy(d_sort, h_sort, sortsize, cudaMemcpyHostToDevice);
    if (cerr) {
        printf("[createData memcpy d_sort] Cuda error: %s\n", cudaGetErrorString(cerr));
        return true;
    }
    if (withId) {
        cerr = cudaMemcpy(d_idx, h_idx, idxsize, cudaMemcpyHostToDevice);
        if (cerr) {
            printf("[createData memcpy d_idx] Cuda error: %s\n", cudaGetErrorString(cerr));
            return true;
        }
    }

    cerr = cudaDeviceSynchronize();
    if (cerr) {
        printf("[createData deviceSync] Cuda error: %s\n", cudaGetErrorString(cerr));
        return true;
    }

    return false;
}

// Helper functions for bit conversions
template<typename T>
inline uint32_t toBitsCpu(T val) {
    if constexpr (std::is_same<T, float>::value) {
        uint32_t fuint;
        memcpy(&fuint, &val, sizeof(float));
        return (fuint & 0x80000000) ? ~fuint : fuint ^ 0x80000000;
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
    uint32_t numScanThreads; // number of scan threads to launch should be multiples of 32 (WARP_SIZE) but based on number of blocks
    uint32_t numDownsweepThreads; // number of downsweep threads multiples of 32 - based on numElemInBlock
    // uint32_t numDownsweepActiveWarps; // number of warps that should participate
    uint32_t downsweepSharedSize; // size of downsweep threads
    uint32_t downsweepKeysPerThread; // number of keys per thread

    uint32_t const numElemInBlock      = 512; // Elements per block
    uint32_t const numUpsweepThreads   = 512; // Num threads per upsweep kernel
    uint32_t const radix = RADIX;
    // uint32_t const downsweepSharedSize = (
    //     (512 + ( WARP_SIZE * BIN_KEYS_PER_THREAD ) -1) / ( WARP_SIZE * BIN_KEYS_PER_THREAD ) * RADIX
    // );

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
        // uint32_t activeDownsweepThreads = ((res.numElemInBlock / BIN_KEYS_PER_THREAD) + LANE_MASK) & ~LANE_MASK;
        // printf("Active downsweep threads: %u\n", activeDownsweepThreads);

        res.numThreadBlocks = (size + res.numElemInBlock - 1) / res.numElemInBlock;
        res.numScanThreads  = (res.numThreadBlocks + LANE_MASK) & ~LANE_MASK;
        res.numDownsweepThreads = 256; // TODO: revisit this
        res.downsweepKeysPerThread = min(res.numElemInBlock/ res.numDownsweepThreads, BIN_KEYS_PER_THREAD);
        res.downsweepSharedSize = (res.numElemInBlock + (WARP_SIZE * res.downsweepKeysPerThread) - 1) / ( WARP_SIZE * res.downsweepKeysPerThread ) * RADIX;
        // res.numDownsweepThreads = activeDownsweepThreads;
        // res.numDownsweepActiveWarps = activeDownsweepThreads / WARP_SIZE;

        return res;
    }
};

template<typename T>
__global__ void printSize(T* d) {
    printf("Size: %lu", sizeof(d));
}

template<typename T, typename U>
uint32_t validate(const uint32_t size, bool dataseq = true, bool withId = false) {
    printf("Validating for size[%u] and typeSize[%lu]\n", size, sizeof(T));
    uint32_t errors = 0;

    Resources res = Resources::compute(size, sizeof(uint32_t));
    printf("For size[%u]\n---------------\nnumThreadBlocks: %u\nnumUpsweepThreads: %u\nnumScanThreads: %u\nnumDownsweepThreads: %u\ndownsweepSharedSize: %u\ndownsweepKeysPerThreade: %u\nmaxNumElementsInBlock: %u\n\n", size, res.numThreadBlocks, res.numUpsweepThreads, res.numScanThreads, res.numDownsweepThreads, res.downsweepSharedSize, res.downsweepKeysPerThread, res.numElemInBlock);
    
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

    c_ret = cudaMalloc(&d_sort, sortSize);
    if (c_ret) {
        errors += 1;
        printf("[Pre-pass -> malloc d_sort] Cuda Error: %s\n", cudaGetErrorString(c_ret));
        return errors;
    }
    c_ret = cudaMalloc(&d_sortAlt, sortSize);
    if (c_ret) {
        errors += 1;
        printf("[Pre-pass -> malloc d_sortAlt] Cuda Error: %s\n", cudaGetErrorString(c_ret));
        return errors;
    }
    c_ret = cudaMalloc(&d_globalHist, radixSize * numPasses);
    if (c_ret) {
        errors += 1;
        printf("[Pre-pass -> malloc d_globalHist] Cuda Error: %s\n", cudaGetErrorString(c_ret));
        return errors;
    }
    c_ret = cudaMalloc(&d_passHist, radixSize * res.numThreadBlocks);
    if (c_ret) {
        errors += 1;
        printf("[Pre-pass -> malloc d_passHist] Cuda Error: %s\n", cudaGetErrorString(c_ret));
        return errors;
    }

    if (withId) {
        c_ret = cudaMalloc(&d_idx, idxSize);
        if (c_ret) {
            errors += 1;
            printf("[Pre-pass -> malloc d_idx] Cuda Error: %s\n", cudaGetErrorString(c_ret));
            return errors;
        }
        c_ret = cudaMalloc(&d_idxAlt, idxSize);
        if (c_ret) {
            errors += 1;
            printf("[Pre-pass -> malloc d_idxAlt] Cuda Error: %s\n", cudaGetErrorString(c_ret));
            return errors;
        }
    }

    // Create some data
    bool iserr = createData<T>(size, d_sort, d_idx, h_sort, h_idx, dataseq, withId);
    if (iserr) {
        errors += 1;
        printf("[Pre-pass -> createData] data creation errored out!\n");
        return errors;
    }

    c_ret = cudaMemset(d_globalHist, 0,  radixSize * numPasses);
    if (c_ret) {
        errors += 1;
        printf("[Pre-pass -> cudaMemset d_globalHist] Cuda error: %s\n", cudaGetErrorString(c_ret));
        return errors;
    }
    c_ret = cudaDeviceSynchronize();
    if (c_ret) {
        errors += 1;
        printf("[Pre-pass -> cudaDeviceSynchronize] Cuda error: %s\n", cudaGetErrorString(c_ret));
        return errors;
    }

    c_ret = cudaGetLastError();
    if (c_ret) {
        errors += 1;
        printf("[Pre-pass] Generic Cuda Error: %s\n", cudaGetErrorString(c_ret));
        return errors;
    }

    for(uint32_t pass=0; pass < numPasses; ++pass) {
    // for(uint32_t pass=0; pass < 1; ++pass) {
        // Run `RadixUpsweep` kernel and validate
        uint32_t shift = pass * 8;
        printf("Pass[%u/ %u] Shift[%u]\n", pass, numPasses - 1, shift);

        {
            // Get current sort data
            T* sortData = (T*)malloc(sortSize);
            c_ret = cudaMemcpy(sortData, d_sort, sortSize, cudaMemcpyDeviceToHost);
            if (c_ret) {
                printf("[Before Upsweep -> cudaMemcpy] Cuda Error: %s\n", cudaGetErrorString(c_ret));
                printSize<<<1, 1>>>(d_sort);
                c_ret = cudaGetLastError();
                if(c_ret) {
                    printf("This shit: %s\n",cudaGetErrorString(c_ret));
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
            for(uint32_t i=0; i<32; ++i) {
                if(std::is_same<T, float>::value)
                    printf("[%u %f] ", i, sortData[i]);
                else
                    printf("[%u %u] ", i, sortData[i]);
            }
            for(uint32_t i=size - 1; i > size - 32; --i) {
                if(std::is_same<T, float>::value)
                    printf("[%u %f] ", i, sortData[i]);
                else
                    printf("[%u %u] ", i, sortData[i]);
            }
            printf("\n");

            RadixUpsweep<T, U><<<res.numThreadBlocks, res.numUpsweepThreads>>>(d_sort, d_globalHist, d_passHist, size, shift, res.numElemInBlock);
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
                uint32_t offset = i * res.numThreadBlocks;
                for(int j=0; j<res.numThreadBlocks; ++j) {
                    cpuPassHist[offset + j] = 0;
                }
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
            // Crea
            for(uint32_t i = 0; i<res.numThreadBlocks; ++i) {
                uint32_t blockstart = i * res.numElemInBlock;
                uint32_t blockend   = min(blockstart + res.numElemInBlock, size);

                for(uint32_t j=blockstart; j < blockend; ++j) {
                    uint32_t bits = toBitsCpu<T>(sortData[j]);
                    bits = ((bits >> shift) & RADIX_MASK);
                    uint32_t trgt = bits * res.numThreadBlocks + i;
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
                        if(errors < 10)
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
        
            RadixScan<<<RADIX, res.numScanThreads, res.numScanThreads * sizeof(uint32_t)>>>(d_passHist, res.numThreadBlocks);
            c_ret = cudaGetLastError();
            if (c_ret) {
                printf("[Scan] Cuda Error: %s", cudaGetErrorString(c_ret));
                errors += 1;
                break;
            }

            uint32_t* passHistGpu = (uint32_t*)malloc(pass_hist_size);
            uint32_t* passHistCpu = (uint32_t*)malloc(pass_hist_size);

            c_ret = cudaMemcpy(passHistGpu, d_passHist, pass_hist_size, cudaMemcpyDeviceToHost);
            if (c_ret) {
                printf("[Scan -> memCpy -> d_passHist] Cuda Error: %s", cudaGetErrorString(c_ret));
                errors += 1;
                break;
            }
            c_ret = cudaDeviceSynchronize();
            if (c_ret) {
                printf("[Scan -> cudaDevSync -> d_passHist] Cuda Error: %s", cudaGetErrorString(c_ret));
                errors += 1;
                break;
            }

            // Create cpu alternate values
            // Process each partition separately
            // For each digit
            for(uint32_t dgt=0; dgt<RADIX; ++dgt) {
                uint32_t sum = 0;
                uint32_t offst = dgt * res.numThreadBlocks;

                for(uint32_t blk=0; blk<res.numThreadBlocks; ++blk) {
                    uint32_t trgt = blk + offst;
                    passHistCpu[trgt] = sum;
                    sum += passHistBefore[trgt];
                }
            }
            
            printf("Validating `passHist` after scan\n");
            bool passHistError = false;
            for (uint32_t digit = 0; digit < RADIX; ++digit) {
                uint32_t offset = digit * res.numThreadBlocks;
                for(uint32_t block=0; block < res.numThreadBlocks; ++block) {
                    uint32_t trgt = offset + block;
                    // if(digit < 6)
                    //     printf("Block[%u] CpuDig[%u]: %u\n", block, digit, passHistCpu[trgt]);
                    if(passHistCpu[trgt] != passHistGpu[trgt]) {
                        errors += 1;
                        passHistError = true;
                        if(errors < 8) {
                            printf("Mismatch at digit %u, block %u: GPU = %u, CPU = %u\n", digit, block, passHistGpu[trgt], passHistCpu[trgt]);
                            printf("Peeking digit %u in passHist: \n", digit);
                            for(uint32_t blk=0; blk<res.numThreadBlocks; ++blk) {
                                uint32_t trg = digit * res.numThreadBlocks + blk;
                                printf("[%u %u] ", blk, passHistBefore[trg]);
                            }
                            printf("\n");
                        }
                    }
                }
            }
            printf("Pass hist validation after `Scan`: %s\n", passHistError ? "FAIL" : "PASS");

            free(passHistBefore);
            free(passHistGpu);
            free(passHistCpu);
        }

        // Finally launch the Downsweep kernel
        {
            uint32_t sharedSize = res.downsweepSharedSize * sizeof(uint32_t); // For the histogram and later keys
                // res.downsweepSharedSize * sizeof(T); // A temp storage for the data

            // RadixDownsweep<<<res.numThreadBlocks, 256>>>(d_sort, d_sortAlt, d_idx, d_idxAlt, d_globalHist, d_passHist, size, shift);
            RadixDownsweep<T, U><<<res.numThreadBlocks, res.numDownsweepThreads, sharedSize>>>(
                d_sort,
                d_sortAlt,
                d_idx,
                d_idxAlt,
                d_globalHist,
                d_passHist,
                size,
                shift,
                res.numElemInBlock,
                res.downsweepSharedSize,
                res.downsweepKeysPerThread,
                withId
            );
            c_ret = cudaGetLastError();
            if (c_ret) {
                printf("[Downsweep] Cuda Error: %s", cudaGetErrorString(c_ret));
                errors += 1;
                break;
            }

            T* sortPass = (T*)malloc(sortSize);
            c_ret = cudaMemcpy(sortPass, d_sortAlt, sortSize, cudaMemcpyDeviceToHost);
            if (c_ret) {
                printf("[Downsweep -> memcpy] Cuda Error: %s", cudaGetErrorString(c_ret));
                errors += 1;
                break;
            }

            c_ret = cudaDeviceSynchronize();
            if (c_ret) {
                printf("[Downsweep - cudaDivSync] Cuda Error: %s", cudaGetErrorString(c_ret));
                errors += 1;
                break;
            }

            printf("Validating `sortAlt` after Downsweep: pass[%u]\n", pass);
            bool passError = false;
            for(uint32_t i=1; i<size; ++i) {
                // Basically at every stage target bits[prev] < bits[current]
                uint32_t prev = toBitsCpu<T>(sortPass[i - 1]) >> shift & RADIX_MASK;
                uint32_t curr = toBitsCpu<T>(sortPass[i]) >> shift & RADIX_MASK;

                if(prev > curr) {
                    if(errors < 32) {
                        printf("Error[%u]@pass[%u]: ", i, pass);
                        if(std::is_same<T, float>::value)
                            printf("Prev[%f %u] This[%f %u]", sortPass[i - 1], prev, sortPass[i], curr);
                        else
                            printf("Prev[%u %u] This[%f %u]", sortPass[i - 1], prev, sortPass[i], curr);
                        printf("\n");
                    }
                    errors += 1;
                    passError = true;
                }
            }
            printf("Sort data validation after `Downsweep` pass[%u]: %s\n", pass, passError ? "FAIL" : "PASS");
            
            free(sortPass);
        }

        if(errors > 0) {
            printf("Breaking at pass %u\n", pass);
            break;
        }

        std::swap(d_sort, d_sortAlt);
        std::swap(d_idx, d_idxAlt);
    }

    // All passes are done!
    // let's do a final validation of the sort data
    T* sorted = (T*)malloc(sortSize);
    c_ret = cudaMemcpy(sorted, d_sort, sortSize, cudaMemcpyDeviceToHost);
    if(c_ret) {
        errors += 1;
        printf("[Final Validation Data copy] Cuda Error: %s", cudaGetErrorString(c_ret));
        return errors;
    }

    uint32_t* sortedIdx;
    if (withId) {
        sortedIdx = (uint32_t*)malloc(idxSize);
        c_ret = cudaMemcpy(sortedIdx, d_idx, idxSize, cudaMemcpyDeviceToHost);
        if (c_ret) {
            errors += 1;
            printf("[Final Validation Idx copy] Cuda Error: %s", cudaGetErrorString(c_ret));
            return errors;
        }
    }

    cudaDeviceSynchronize();

    printf("Validating final sort data with Indices: %s\n", withId ? "TRUE" : "FALSE");
    bool isSorted = true;
    for(uint32_t idx=1; idx<size; ++idx) {
        if(
            sorted[idx - 1] > sorted[idx] ||
            withId ? ( sorted[idx] != h_sort[sortedIdx[idx]] ) : false
        ) {
            isSorted = false;
            errors += 1;
            if(errors < 16) {
                printf("Unsorted[%u]: ", idx);
                if(std::is_same<T, float>::value)
                    printf("Prev[%f] This[%f]", sorted[idx - 1], sorted[idx]);
                else
                    printf("Prev[%u] This[%u]", sorted[idx - 1], sorted[idx]);
                printf("\n");
            }
        }

        // a test of stability of this sort!
        if (withId) {
            if (sorted[idx - 1] == sorted[idx] && sortedIdx[idx - 1] >= sortedIdx[idx]) {
                printf("Stability Issue[%u]: Prev[%u] Now[%u]\n", idx, sortedIdx[idx - 1], sortedIdx[idx]);
            }
        }
    }
    printf("Final sort validation: `%s`\n", isSorted ? "PASS" : "FAIL");
    printf("=========================================================\n");

    cudaFree(d_sort);
    cudaFree(d_sortAlt);
    cudaFree(d_globalHist);
    cudaFree(d_passHist);

    free(h_sort);
    free(h_idx);
    free(sorted);

    if (withId) {
        cudaFree(d_idx);
        cudaFree(d_idxAlt);
        free(sortedIdx);
    }

    return errors;
}

int main() {
    const uint32_t N = 13;
    uint32_t sizes[N] = { 1024, 1120, 2048, 4096, 4113, 7680, 8192, 9216, 16000, 32000, 64000, 128000, 280000 };
    
    // First, test for UpsweepKernel is good?
    for(uint32_t i = 0; i < N; i++) {
    // for(uint32_t i = 0; i < 1; i++) {
        {
            printf("`uint32_t`: Validation (sequential)\n");
            uint32_t errors = validate<uint32_t, uint32_t>(sizes[i]);
            if(errors > 0){
                printf("Errors: %u while validating for size[uint32_t][%u]\n", errors, sizes[i]);
                break;
            }
        }

        {
            printf("`uint32_t`: Validation (sequential argsort)\n");
            uint32_t errors = validate<uint32_t, uint32_t>(sizes[i], true, true);
            if(errors > 0){
                printf("Errors: %u while validating for size[uint32_t][%u]\n", errors, sizes[i]);
                break;
            }
        }

        {
            printf("`uint32_t`: Validation (random)\n");
            uint32_t errors = validate<uint32_t, uint32_t>(sizes[i], false);
            if(errors > 0) {
                printf("Errors: %u while validating for size[uint32_t][%u]\n", errors, sizes[i]);
                break;
            }
        }

        {
            printf("`uint32_t`: Validation (random argsort)\n");
            uint32_t errors = validate<uint32_t, uint32_t>(sizes[i], false, true);
            if(errors > 0) {
                printf("Errors: %u while validating for size[uint32_t][%u]\n", errors, sizes[i]);
                break;
            }
        }

        {
            printf("`float`: Validation (sequential)\n");
            uint32_t errors = validate<float, uint32_t>(sizes[i]);
            if(errors > 0) {
                printf("Errors: %u while validating for size[float][%u]\n", errors, sizes[i]);
                break;
            }
        }

        {
            printf("`float`: Validation (sequential argsort)\n");
            uint32_t errors = validate<float, uint32_t>(sizes[i], true, true);
            if(errors > 0) {
                printf("Errors: %u while validating for size[float][%u]\n", errors, sizes[i]);
                break;
            }
        }

        {
            printf("`float`: Validation (random)\n");
            uint32_t errors = validate<float, uint32_t>(sizes[i], false);
            if(errors > 0) {
                printf("Errors: %u while validating for size[float][%u]\n", errors, sizes[i]);
                break;
            }
        }

        {
            printf("`float`: Validation (random argsort)\n");
            uint32_t errors = validate<float, uint32_t>(sizes[i], false, true);
            if(errors > 0) {
                printf("Errors: %u while validating for size[float][%u]\n", errors, sizes[i]);
                break;
            }
        }

        {
            printf("`uint8_t`: Validation (random)\n");
            uint32_t errors = validate<uint8_t, uint8_t>(sizes[i], false);
            if(errors > 0) {
                printf("Errors: %u while validating for size[uint8_t][%u]\n", errors, sizes[i]);
                break;
            }
        }

        {
            printf("`uint8_t`: Validation (random argsort)\n");
            uint32_t errors = validate<uint8_t, uint8_t>(sizes[i], false, true);
            if(errors > 0) {
                printf("Errors: %u while validating for size[uint8_t][%u]\n", errors, sizes[i]);
                break;
            }
        }

        // {
        //     printf("`float16`: Validation (seequential)\n");
        //     uint32_t errors = validateUpsweep<__fp16>(sizes[i], false);
        //     if(errors > 0)
        //         printf("Errors: %u while validating for size[fp16][%u]\n", errors, sizes[i]);
        // }

        // {
        //     printf("`float16`: Validation (seequential)\n");
        //     uint32_t errors = validateUpsweep<__nv_bfloat16>(sizes[i], false);
        //     if(errors > 0)
        //         printf("Errors: %u while validating for size[bfloat16][%u]\n", errors, sizes[i]);
        // }
        // {
        //     uint32_t errors = validateUpsweep<half>(sizes[i]);
        //     if(errors > 0)
        //         printf("Errors: %u while validating for size[float16][%u]", errors, sizes[i]);
        // }
    }
    return 0;
}