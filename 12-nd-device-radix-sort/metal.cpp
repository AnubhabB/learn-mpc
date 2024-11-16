#include <stdint.h>
#include <stdio.h>

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>

using namespace std;

#define RADIX 256 // 8-bit RADIX
#define RADIX_MASK (RADIX - 1)
#define SIMD_SIZE 32 // This can change, can we visit this later?
#define LANE_MASK (SIMD_SIZE - 1)

#define BIN_KEYS_PER_THREAD 15

// Load the `.metal` kernel definition as a string
static const char* kernel = {
    #include "radix.metal"
};

// Random float helper
static inline float random_float() {
    return (float)rand() / (float)RAND_MAX;
}

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

template<>
inline float random_range(float min, float max) {
    return min + random_float() * (max - min);
}

// Specialization for half float if needed
template<>
inline __fp16 random_range(__fp16 min, __fp16 max) {
    float min_f = float(min);
    float max_f = float(max);
    return __fp16(min_f + random_float() * (max_f - min_f));
}

// Specialization for half float if needed
// template<>
// inline __bf16 random_range(__bf16 min, __bf16 max) {
//     float min_f = float(min);
//     float max_f = float(max);
//     return __bf16(min_f + random_float() * (max_f - min_f));
// }

// Helper functions for bit conversions
template<typename T, typename U>
inline U toBitsCpu(T val) {
    if constexpr (std::is_same<T, float>::value) {
        uint32_t fuint;
        memcpy(&fuint, &val, sizeof(float));
        return (fuint & 0x80000000) ? ~fuint : fuint ^ 0x80000000;
    } else if constexpr (std::is_same<T, __fp16>::value) {
        if (std::isfinite(static_cast<float>(val))) {
            ushort bits = *reinterpret_cast<ushort*>(&val);
            return static_cast<ushort>((bits & 0x8000) ? ~bits : bits ^ 0x8000);  // 0x8000 is sign bit for 16-bit
        }

        return isnan(val) || val > 0.0 ? 0xFFFF : 0;
    // } else if constexpr (std::is_same<T, __bf16>::value) {
    //     if (std::isfinite(static_cast<float>(val))) {
    //         ushort bits = *reinterpret_cast<ushort*>(&val);
    //         return static_cast<ushort>((bits & 0x8000) ? ~bits : bits ^ 0x8000);  // 0x8000 is sign bit for 16-bit
    //     }

    //     return isnan(val) || val > 0.0 ? 0xFFFF : 0;
    } else {
        return static_cast<U>(val);
    }
}

template<typename T>
bool createData(uint32_t size, T* d_sort, uint32_t* d_idx, bool seq, bool withId) {
    uint32_t sortsize = size * sizeof(T);
    uint32_t idxsize  = withId ? size * sizeof(uint32_t) : 0;

    T min;
    T max;

    if(!seq) {
        if constexpr (std::is_same<T, uint8_t>::value) {
            min = (uint8_t)0;
            max = (uint8_t)255;
        } else if constexpr (std::is_same<T, uint32_t>::value) {
            min = (uint32_t)0;
            max = size <= 1024 ? 64 : (uint32_t)320000;
        } else if constexpr (std::is_same<T, float>::value) {
            min = -128000.0f;
            max = 128000.0f;
        } else if constexpr (std::is_same<T, __fp16>::value) {
            min = -24000.0;
            max = 24000.0;
        // } else if constexpr (std::is_same<T, __bf16>::value) {
        //     min = -255.0;
        //     max = 255.0;;
        //     max = __float2bfloat16(6000000.0f);
        // } else if constexpr (std::is_same<T, int64_t>::value) {
        //     min = INT64_MIN;
        //     max = INT64_MAX;
        }
    }

    for(uint32_t i=0; i<size; i++) {
        if(seq) {
            d_sort[i] = static_cast<T>(i);
        } else {
            d_sort[i] = random_range(min, max);
        }
        if (withId)
            d_idx[i] = i;
    }

    return false;
}

// Calculate resources to run
struct Resources {
    uint32_t numPartitions;          // number of threadblocks to run for Upsweep and DownsweepPairs kernel
    uint32_t numElemInPartition;     // number of elements per partition
    uint32_t numScanThreads;         // number of threads to launch the scan kernel
    uint32_t numDownsweepThreads;    // number of downsweep threads
    uint32_t downsweepKeysPerThread; // number of keys to be processed by one downsweep thread capped at BIN_KEYS_PER_THREAD
    uint32_t downsweepSharedSize;    // shared memory size of downsweep threads

    static Resources compute(MTL::Device* device, uint32_t size, uint32_t type_size) {
        Resources resc;

        uint32_t avlUpsweepSharedMem = ((device->maxThreadgroupMemoryLength() - RADIX * sizeof(uint32_t)) * 3)/ 5;

        resc.numElemInPartition  = min(size & ~31, 1024u);

        resc.numPartitions   = (size + resc.numElemInPartition - 1) / resc.numElemInPartition;
        resc.numScanThreads  = (resc.numPartitions + LANE_MASK) & ~LANE_MASK;

        resc.numDownsweepThreads    = 512;
        resc.downsweepKeysPerThread = min((resc.numElemInPartition  + resc.numDownsweepThreads - 1)/ resc.numDownsweepThreads, (uint32_t)BIN_KEYS_PER_THREAD);
        resc.downsweepSharedSize    = (resc.numElemInPartition + (SIMD_SIZE * resc.downsweepKeysPerThread) - 1) / ( SIMD_SIZE * resc.downsweepKeysPerThread ) * RADIX;

        return resc;
    }
};

template<typename T, typename U>
uint32_t validate(const uint32_t size, bool dataseq = true, bool withId = false) {
    printf("Validating for size[%u] and typeSize[%lu]\n", size, sizeof(T));
    uint32_t errors = 0;

    // Get the device
    MTL::Device* device = MTLCreateSystemDefaultDevice();
    NS::Error* pError = nullptr;

    Resources resc = Resources::compute(device, size, sizeof(uint32_t));

    const uint32_t numPasses  = sizeof(T);
    const uint32_t sortSize   = size * sizeof(T);
    const uint32_t idxSize    = withId ? size * sizeof(uint32_t) : 0;
    const uint32_t radixSize  = RADIX * sizeof(uint32_t);

    T* h_sort       = (T*)malloc(sortSize);
    uint32_t* h_idx = (uint32_t*)malloc(idxSize);
    
    // Create a library and lookup the kernel function
    MTL::Library* library    = device->newLibrary( NS::String::string(kernel, NS::UTF8StringEncoding), nullptr, &pError );
    if (pError) {
        errors += 1;
        __builtin_printf( "Library load error: %s\n", pError->localizedDescription()->utf8String() );
        __builtin_printf( "Library load error(debug): %s\n", pError->debugDescription()->utf8String() );
        return errors;
    }

    // kernel names
    char upsweepFn[32];
    char downsweepFn[36];
    if constexpr (std::is_same<T, uint8_t>::value) {
        // upFn = upFn + "uint8_t_uint8_t";
        strcpy(upsweepFn, "RadixUpsweep_uint8_t_uint8_t");
        strcpy(downsweepFn, "RadixDownsweep_uint8_t_uint8_t");
    } else if constexpr (std::is_same<T, uint32_t>::value) {
        strcpy(upsweepFn, "RadixUpsweep_uint32_t_uint32_t");
        strcpy(downsweepFn, "RadixDownsweep_uint32_t_uint32_t");
    } else if constexpr (std::is_same<T, float>::value) {
        strcpy(upsweepFn, "RadixUpsweep_float_uint32_t");
        strcpy(downsweepFn, "RadixDownsweep_float_uint32_t");
    } else if constexpr (std::is_same<T, __fp16>::value) {
        strcpy(upsweepFn, "RadixUpsweep_half_ushort");
        strcpy(downsweepFn, "RadixDownsweep_half_ushort");
    // } else if constexpr (std::is_same<T, __bf16>::value) {
    //     strcpy(upsweepFn, "RadixUpsweep_bfloat_ushort");
    //     strcpy(downsweepFn, "RadixDownsweep_bfloat_ushort");
    }

    MTL::Function* _up   = library->newFunction( NS::String::string(upsweepFn, NS::UTF8StringEncoding) );
    MTL::Function* _scan = library->newFunction( NS::String::string("RadixScan", NS::UTF8StringEncoding) );
    MTL::Function* _down = library->newFunction( NS::String::string(downsweepFn, NS::UTF8StringEncoding) );

    if(!_up) {
        errors += 1;
        __builtin_printf( "UpsweepKernel: function creation error: %s\n", pError->localizedDescription()->utf8String() );
        __builtin_printf( "UpsweepKernel: function creation error(debug): %s\n", pError->debugDescription()->utf8String() );
        return errors;
    }

    // Declarations
    // Array buffers first
    MTL::Buffer* d_sort       = device->newBuffer(sortSize, MTL::ResourceStorageModeShared);
    MTL::Buffer* d_sortAlt    = device->newBuffer(sortSize, MTL::ResourceStorageModeShared);
    MTL::Buffer* d_globalHist = device->newBuffer(radixSize * numPasses, MTL::ResourceStorageModeShared);
    MTL::Buffer* d_passHist   = device->newBuffer(radixSize * resc.numPartitions, MTL::ResourceStorageModeShared);
    MTL::Buffer* d_idx        = device->newBuffer(idxSize, MTL::ResourceStorageModeShared);
    MTL::Buffer* d_idxAlt     = device->newBuffer(idxSize, MTL::ResourceStorageModeShared);

    // Other stuff
    MTL::Buffer* d_size          = device->newBuffer(&size, sizeof(uint32_t), MTL::ResourceStorageModePrivate);
    MTL::Buffer* radixShift      = device->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    MTL::Buffer* elemsPerPart    = device->newBuffer(&resc.numElemInPartition, sizeof(uint32_t), MTL::ResourceStorageModePrivate);
    MTL::Buffer* numParts        = device->newBuffer(&resc.numPartitions, sizeof(uint32_t), MTL::ResourceStorageModePrivate);
    MTL::Buffer* d_histsSize     = device->newBuffer(&resc.downsweepSharedSize, sizeof(uint32_t), MTL::ResourceStorageModePrivate);
    MTL::Buffer* d_keysPerThread = device->newBuffer(&resc.downsweepKeysPerThread, sizeof(uint32_t), MTL::ResourceStorageModePrivate);
    MTL::Buffer* d_sortIdx       = device->newBuffer(&withId, sizeof(bool), MTL::ResourceStorageModePrivate);

    // Create some data
    T* sort_buf       = static_cast<T*>(d_sort->contents());
    uint32_t* idx_buf = static_cast<uint32_t*>(d_idx->contents());

    if (createData<T>(size, sort_buf, idx_buf, dataseq, withId)) {
        errors += 1;
        printf("[Pre-pass -> createData] data creation errored out!\n");
        return errors;
    }

    T* unsorted = (T*)malloc(sortSize);
    memcpy(unsorted, sort_buf, sortSize);

    // Set global histogram to zero
    void* gHist = d_globalHist->contents();
    std::memset(gHist, 0, d_globalHist->length());

    // Create a command queue
    MTL::CommandQueue *cmdQueue = device->newCommandQueue();
    // Create pipeline states
    MTL::ComputePipelineState *_upsweepState = device->newComputePipelineState(_up, &pError);
    if (pError) {
        errors += 1;
        __builtin_printf( "RadixUpsweep: state initialization failed: %s\n", pError->localizedDescription()->utf8String() );
        return errors;
    }
    MTL::ComputePipelineState *_scanState = device->newComputePipelineState(_scan, &pError);
    if (pError) {
        errors += 1;
        __builtin_printf( "RadixScan: state initialization failed: %s", pError->localizedDescription()->utf8String() );
        return errors;
    }
    MTL::ComputePipelineState *_downsweepState = device->newComputePipelineState(_down, &pError);
    if (pError) {
        errors += 1;
        __builtin_printf( "RadixDownsweep: state initialization failed: %s", pError->localizedDescription()->utf8String() );
        return errors;
    }

    ushort _upExeWidth = _upsweepState->threadExecutionWidth();
    ushort _nUpThreads  = min((uint32_t)(((size + resc.numPartitions - 1) / resc.numPartitions + _upExeWidth) & ~_upExeWidth), (uint32_t)device->maxThreadsPerThreadgroup().width);
    
    MTL::Size upThreadsPerGroup   = MTL::Size::Make(_nUpThreads, 1, 1);
    MTL::Size upDownGroupsPerGrid = MTL::Size::Make(resc.numPartitions, 1, 1);

    MTL::Size scanThreadsPerGroup = MTL::Size::Make(resc.numScanThreads, 1, 1);
    MTL::Size scanGroupsPerGrid   = MTL::Size::Make(RADIX, 1, 1);

    MTL::Size downThreadsPerGroup = MTL::Size::Make(resc.numDownsweepThreads, 1, 1);

    printf("For size[%u]\n---------------\nnumPartitions: %u\nnumUpsweepThreads: %u\nnumScanThreads: %u\nnumDownsweepThreads: %u\ndownsweepSharedSize: %u\ndownsweepKeysPerThreade: %u\nmaxNumElementsInBlock: %u\n\n", size, resc.numPartitions, _nUpThreads, resc.numScanThreads, resc.numDownsweepThreads, resc.downsweepSharedSize, resc.downsweepKeysPerThread, resc.numElemInPartition);

    // MTL::CommandBuffer* cmdBuffer = cmdQueue->commandBuffer();

    uint32_t* shift = static_cast<uint32_t*>(radixShift->contents());
    for (uint32_t pass = 0; pass < numPasses; ++pass) {
    // for (uint32_t pass = 1; pass < 2; ++pass) {
        MTL::CommandBuffer* cmdBuffer = cmdQueue->commandBuffer();
        *shift = pass * 8;
        printf("Pass[%u/ %u] Shift[%u]\n", pass, numPasses - 1, *shift);

        // The upsweep kernel call
        {
            // Create a command buffer and a compute encoder from the buffer
            // MTL::CommandBuffer* cmdBuffer = cmdQueue->commandBuffer();

            // Get current data being sorted
            // T* sortData = static_cast<T*>(d_sort->contents());
            // printf("Sort data now: \n");
            // for(uint32_t i=0; i<32; ++i) {
            //     if constexpr (std::is_same<T, float>::value)
            //         printf("[%u %f] ", i, sortData[i]);
            //     // else if constexpr (std::is_same<T, half>::value)
            //     //     printf("[%u %f] ", i, __half2float(sortData[i]));
            //     // else if constexpr (std::is_same<T, nv_bfloat16>::value)
            //     //     printf("[%u %f] ", i, __bfloat162float(sortData[i]));
            //     else
            //         printf("[%u %u] ", i, sortData[i]);
            // }
            // printf(" ... ... ");
            // for(uint32_t i=size - 32; i < size; ++i) {
            //     if constexpr (std::is_same<T, float>::value)
            //         printf("[%u %f] ", i, sortData[i]);
            //     // else if constexpr (std::is_same<T, half>::value)
            //     //     printf("[%u %f] ", i, __half2float(sortData[i]));
            //     // else if constexpr (std::is_same<T, nv_bfloat16>::value)
            //     //     printf("[%u %f] ", i, __bfloat162float(sortData[i]));
            //     else
            //         printf("[%u %u] ", i, sortData[i]);
            // }
            // printf("\n");

            MTL::ComputeCommandEncoder* upsweepEncoder = cmdBuffer->computeCommandEncoder();
            upsweepEncoder->setComputePipelineState(_upsweepState);
            upsweepEncoder->setBuffer(d_sort, 0, 0);
            upsweepEncoder->setBuffer(d_globalHist, 0, 1);
            upsweepEncoder->setBuffer(d_passHist, 0, 2);
            upsweepEncoder->setBuffer(d_size, 0, 3);
            upsweepEncoder->setBuffer(radixShift, 0, 4);
            upsweepEncoder->setBuffer(elemsPerPart, 0, 5);

            upsweepEncoder->setThreadgroupMemoryLength(RADIX * sizeof(uint32_t), 0);

            upsweepEncoder->dispatchThreadgroups(upDownGroupsPerGrid, upThreadsPerGroup);
            upsweepEncoder->endEncoding();

            // // Commit and wait for completion
            // cmdBuffer->commit();
            // cmdBuffer->waitUntilCompleted();

            // uint32_t *cpuGlobHist = (uint32_t*)malloc(radixSize);
            // uint32_t *gpuGlobHist = static_cast<uint32_t*>(d_globalHist->contents());
            // uint32_t *cpuPassHist = (uint32_t*)malloc(radixSize * resc.numPartitions);
            // uint32_t *gpuPassHist = static_cast<uint32_t*>(d_passHist->contents());

            // Initialize all zeroes
            // for(int i=0; i<RADIX; ++i) {
            //     cpuGlobHist[i] = 0;
            //     uint32_t offset = i * resc.numPartitions;
            //     for(int j=0; j<resc.numPartitions; ++j) {
            //         cpuPassHist[offset + j] = 0;
            //     }
            // }

            // // Compute CPU histogram
            // for (uint32_t i = 0; i < size; i++) {
            //     uint32_t bits = toBitsCpu<T, U>(sortData[i]);
            //     uint32_t digit = (bits >> *shift) & RADIX_MASK;
            //     cpuGlobHist[digit]++;
            // }
            // // Convert to exclusive prefix sum
            // uint32_t prev = 0;
            // for (uint32_t i = 0; i < RADIX; i++) {
            //     uint32_t current = cpuGlobHist[i];
            //     cpuGlobHist[i] = prev;
            //     prev += current;
            // }
            // // Crea
            // for(uint32_t i = 0; i<resc.numPartitions; ++i) {
            //     uint32_t blockstart = i * resc.numElemInPartition;
            //     uint32_t blockend   = min(blockstart + resc.numElemInPartition, size);

            //     for(uint32_t j=blockstart; j < blockend; ++j) {
            //         uint32_t bits = toBitsCpu<T, U>(sortData[j]);
            //         bits = ((bits >> *shift) & RADIX_MASK);
            //         uint32_t trgt = bits * resc.numPartitions + i;
            //         cpuPassHist[trgt] += 1;
            //     }
            // }

            // bool passHistError = false;
            // for(uint32_t block=0; block < resc.numPartitions; ++block) {
            //     uint32_t offset = block * RADIX;

            //     for(uint32_t digit=0; digit < RADIX; digit++) {
            //         uint32_t v_idx = offset + digit;
            //         if(cpuPassHist[v_idx] != gpuPassHist[v_idx]) {
            //             if(!passHistError)
            //                 printf("Validating `passHist`\n");
            //             errors++;
            //             passHistError = true;
            //             if(errors < 10)
            //                 printf("Error @ Block[%u] Digit[%u]: Cpu[%u] Gpu[%u]\n", block, digit, cpuPassHist[v_idx], gpuPassHist[v_idx]);
            //         }
            //     }
            // }
            // printf("Pass hist validation: %s\n", passHistError ? "FAIL" : "PASS");

            // bool globalHistError = false;
            // uint32_t* gpuCurrent = gpuGlobHist + RADIX * pass;
            // for(uint32_t i=0; i<RADIX; ++i) {
            //     if(cpuGlobHist[i] != gpuCurrent[i]) {
            //         if(!globalHistError)
            //             printf("Validating `globalHist`\n");
            //         errors++;
            //         globalHistError = true;
            //         if(errors < 32)
            //             printf("Error @ Digit[%u]: Cpu[%u] Gpu[%u]\n", i, cpuGlobHist[i], gpuGlobHist[i]);
            //     }
            // }
            // printf("Global hist validation: %s\n", globalHistError ? "FAIL" : "PASS");
            
            // free(cpuGlobHist);
            // free(cpuPassHist);
        }

        // Launch RadixScan kernel
        {
            uint32_t pass_hist_size = radixSize * resc.numPartitions;

            // Create a command buffer and a compute encoder from the buffer
            // MTL::CommandBuffer* cmdBuffer = cmdQueue->commandBuffer();

            // uint32_t* passHistGpu = static_cast<uint32_t*>(d_passHist->contents());
            // uint32_t* passHistBefore = (uint32_t*)malloc(pass_hist_size);
            // memcpy(passHistBefore, passHistGpu, pass_hist_size);

            // printf("\nBEFORE ==================\n");
            // for(uint32_t i=0; i<32; ++i) {
            //     printf("[%u %u] ", i, passHistGpu[i]);
            // }
            // printf("\n========================\n");

            MTL::ComputeCommandEncoder* scanEncoder = cmdBuffer->computeCommandEncoder();
            scanEncoder->setComputePipelineState(_scanState);
            scanEncoder->setBuffer(d_passHist, 0, 0);
            scanEncoder->setBuffer(numParts, 0, 1);

            scanEncoder->setThreadgroupMemoryLength(resc.numScanThreads * sizeof(uint32_t), 0);

            scanEncoder->dispatchThreadgroups(scanGroupsPerGrid, scanThreadsPerGroup);
            scanEncoder->endEncoding();

            // // Commit and wait for completion
            // cmdBuffer->commit();
            // cmdBuffer->waitUntilCompleted();

            // // printf("\nAFTER ==================\n");
            // // for(uint32_t i=0; i<32; ++i) {
            // //     printf("[%u %u] ", i, passHistGpu[i]);
            // // }
            // // printf("\n========================\n");
            // uint32_t* passHistCpu = (uint32_t*)malloc(pass_hist_size);
            // // Create cpu alternate values
            // // Process each partition separately
            // // For each digit
            // for(uint32_t dgt=0; dgt<RADIX; ++dgt) {
            //     uint32_t sum = 0;
            //     uint32_t offst = dgt * resc.numPartitions;

            //     for(uint32_t blk=0; blk<resc.numPartitions; ++blk) {
            //         uint32_t trgt = blk + offst;
            //         passHistCpu[trgt] = sum;
            //         sum += passHistBefore[trgt];
            //     }
            // }

            // bool passHistError = false;
            // for (uint32_t digit = 0; digit < RADIX; ++digit) {
            //     uint32_t offset = digit * resc.numPartitions;
            //     for(uint32_t block=0; block < resc.numPartitions; ++block) {
            //         uint32_t trgt = offset + block;
            //         // if(digit < 6)
            //         //     printf("Block[%u] CpuDig[%u]: %u\n", block, digit, passHistCpu[trgt]);
            //         if(passHistCpu[trgt] != passHistGpu[trgt]) {
            //             if(!passHistError) {
            //                 printf("Validating `passHist` after scan\n");
            //             }
            //             errors += 1;
            //             passHistError = true;
            //             if(errors < 16) {
            //                 printf("Mismatch at digit %u, block %u: GPU = %u, CPU = %u\n", digit, block, passHistGpu[trgt], passHistCpu[trgt]);
            //                 printf("Peeking digit %u in passHist: \n", digit);
            //                 for(uint32_t blk=0; blk<resc.numPartitions; ++blk) {
            //                     uint32_t trg = digit * resc.numPartitions + blk;
            //                     printf("[%u %u] ", blk, passHistBefore[trg]);
            //                 }
            //                 printf("\n");
            //             }
            //         }
            //     }
            // }
            // printf("Pass hist validation after `Scan`: %s\n", passHistError ? "FAIL" : "PASS");

            // free(passHistBefore);
            // free(passHistCpu);
        }

        // Finally the downsweep kernel
        {
            // Create a command buffer and a compute encoder from the buffer
            // MTL::CommandBuffer* cmdBuffer = cmdQueue->commandBuffer();
            MTL::ComputeCommandEncoder* downEncoder = cmdBuffer->computeCommandEncoder();
            downEncoder->setComputePipelineState(_downsweepState);
            downEncoder->setBuffer(d_sort, 0, 0);
            downEncoder->setBuffer(d_sortAlt, 0, 1);
            downEncoder->setBuffer(d_idx, 0, 2);
            downEncoder->setBuffer(d_idxAlt, 0, 3);
            downEncoder->setBuffer(d_globalHist, 0, 4);
            downEncoder->setBuffer(d_passHist, 0, 5);
            downEncoder->setBuffer(d_size, 0, 6);
            downEncoder->setBuffer(radixShift, 0, 7);
            downEncoder->setBuffer(elemsPerPart, 0, 8);
            downEncoder->setBuffer(d_histsSize, 0, 9);
            downEncoder->setBuffer(d_keysPerThread, 0, 10);
            downEncoder->setBuffer(d_sortIdx, 0, 11);

            downEncoder->setThreadgroupMemoryLength(resc.downsweepSharedSize * sizeof(uint32_t), 0);
            downEncoder->setThreadgroupMemoryLength(RADIX * sizeof(uint32_t), 1);

            downEncoder->dispatchThreadgroups(upDownGroupsPerGrid, downThreadsPerGroup);
            downEncoder->endEncoding();

            /****************
             * 
             * Uncomment the following for validating the `downsweep` pass
             * 
            ****************/
            // cmdBuffer->commit();
            // cmdBuffer->waitUntilCompleted();

            // T* sortPass = static_cast<T*>(d_sortAlt->contents());
            // uint32_t* idxPass = static_cast<uint32_t*>(d_idxAlt->contents());

            // for(uint32_t i=0; i<size; ++i) {
            //     printf("[%u %f] ", i, sortPass[i]);
            // }

            // bool passError = false;
            // for(uint32_t i=1; i<size; ++i) {
            //     // Basically at every stage target bits[prev] < bits[current]
            //     U prev = toBitsCpu<T, U>(sortPass[i - 1]) >> *shift & RADIX_MASK;
            //     U curr = toBitsCpu<T, U>(sortPass[i]) >> *shift & RADIX_MASK;

            //     if(prev > curr) {
            //         if(!passError) {
            //             printf("Validating `sortAlt` after Downsweep: pass[%u]\n", pass);
            //         }
            //         if(errors < 32) {
            //             printf("Error[%u]@pass[%u]: ", i, pass);
            //             if constexpr (std::is_same<T, float>::value)
            //                 printf("Prev[%f %u] This[%f %u]", sortPass[i - 1], prev, sortPass[i], curr);
            //             // else if constexpr (std::is_same<T, half>::value)
            //             //     printf("Prev[%f %u] This[%f %u]", __half2float(sortPass[i - 1]), prev, __half2float(sortPass[i]), curr);
            //             // else if constexpr (std::is_same<T, nv_bfloat16>::value)
            //             //     printf("Prev[%f %u] This[%f %u]", __bfloat162float(sortPass[i - 1]), prev, __bfloat162float(sortPass[i]), curr);
            //             else if constexpr (std::is_same<T, uint32_t>::value)
            //                 printf("Prev[%u %u] This[%u %u]", sortPass[i - 1], prev, sortPass[i], curr);
            //             else if constexpr (std::is_same<T, uint8_t>::value)
            //                 printf("Prev[%u %u] This[%u %u]", sortPass[i - 1], prev, sortPass[i], curr);
            //             // else if constexpr (std::is_same<T, int64_t>::value)
            //             //     printf("Prev[%ld %u] This[%ld %u]", (long)sortPass[i - 1], prev, (long)sortPass[i], curr);
            //             printf("\n");
            //         }
            //         errors += 1;
            //         passError = true;
            //     }
            // }
            // printf("Sort data validation after `Downsweep` pass[%u]: %s\n", pass, passError ? "FAIL" : "PASS");
            /****************
             * 
             * End of `downsweep` validation
             * 
            ****************/
        }

        cmdBuffer->commit();
        cmdBuffer->waitUntilCompleted();

        if(errors > 0) {
            printf("Breaking at pass %u\n", pass);
            break;
        }

        std::swap(d_sort, d_sortAlt);
        if (withId)
            std::swap(d_idx, d_idxAlt);
    }

    // Commit and wait for completion
    // cmdBuffer->commit();
    // cmdBuffer->waitUntilCompleted();

    // All passes are done!
    // let's do a final validation of the sort data
    T* sorted = static_cast<T*>(d_sort->contents());
    uint32_t* sortedIdx = static_cast<uint32_t*>(d_idx->contents());

    printf("Validating final sort data with Indices: %s\n", withId ? "TRUE" : "FALSE");
    bool isSorted = true;
    for(uint32_t idx=1; idx<size; ++idx) {
        if(
            (!dataseq && sorted[idx - 1] > sorted[idx]) ||
            (dataseq && sorted[idx - 1] >= sorted[idx]) ||
            withId ? ( sorted[idx] != unsorted[sortedIdx[idx]] ) : false
        ) {
            isSorted = false;
            errors += 1;
            if(errors < 16) {
                printf("Unsorted[%u]: ", idx);
                if constexpr (std::is_same<T, float>::value) {
                    printf("Prev[%f] This[%f]", sorted[idx - 1], sorted[idx]);
                    if (withId)
                        printf(" Original[%u][%f]", sortedIdx[idx], unsorted[sortedIdx[idx]]);
                else if constexpr (std::is_same<T, __fp16>::value)
                    printf("Prev[%f] This[%f]", sorted[idx - 1], sorted[idx]);
                else if constexpr (std::is_same<T, __bf16>::value)
                    printf("Prev[%f] This[%f]", sorted[idx - 1], sorted[idx]);
                } else if constexpr (std::is_same<T, uint32_t>::value) {
                    printf("Prev[%u] This[%u]", sorted[idx - 1], sorted[idx]);
                    if (withId)
                        printf(" Original[%u][%u]", sortedIdx[idx], unsorted[sortedIdx[idx]]);
                } // else if constexpr (std::is_same<T, int64_t>::value)
                //     printf("Prev[%ld] This[%ld]", (long)sorted[idx - 1], (long)sorted[idx]);
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

    // Cleanup buffers
    if (d_sort)
        d_sort -> release();
    if (d_sortAlt)
        d_sortAlt -> release();
    if (d_idx)
        d_idx -> release();
    if (d_idxAlt)
        d_idxAlt -> release();
    if (d_globalHist)
        d_globalHist -> release();
    if (d_passHist)
        d_passHist -> release();
    
    // if (cmdBuffer)
    //     cmdBuffer -> release();
    if (cmdQueue)
        cmdQueue -> release();

    if (_upsweepState)
        _upsweepState -> release();
    if (_scanState)
        _scanState -> release();
    if (_downsweepState)
        _downsweepState -> release();

    if (library)
        library -> release();

    return errors;
}

int main() {
    const uint32_t N = 13;
    uint32_t sizes[N] = { 1024, 1324, 2048, 4096, 4113, 7680, 8192, 9216, 16000, 32000, 64000, 128000, 280000 };
    
    // First, test for UpsweepKernel is good?
    for(uint32_t i = 0; i < N; i++) {
    // for(uint32_t i = 1; i < 2; i++) {
        // {
        //     printf("`uint32_t`: Validation (sequential)\n");
        //     uint32_t errors = validate<uint32_t, uint32_t>(sizes[i]);
        //     if(errors > 0){
        //         printf("Errors: %u while validating for size[uint32_t][%u]\n", errors, sizes[i]);
        //         break;
        //     }
        // }

        // {
        //     printf("`uint32_t`: Validation (sequential argsort)\n");
        //     uint32_t errors = validate<uint32_t, uint32_t>(sizes[i], true, true);
        //     if(errors > 0){
        //         printf("Errors: %u while validating for size[uint32_t][%u]\n", errors, sizes[i]);
        //         break;
        //     }
        // }

        // {
        //     printf("`uint32_t`: Validation (random)\n");
        //     uint32_t errors = validate<uint32_t, uint32_t>(sizes[i], false);
        //     if(errors > 0) {
        //         printf("Errors: %u while validating for size[uint32_t][%u]\n", errors, sizes[i]);
        //         break;
        //     }
        // }

        // {
        //     printf("`uint32_t`: Validation (random argsort)\n");
        //     uint32_t errors = validate<uint32_t, uint32_t>(sizes[i], false, true);
        //     if(errors > 0) {
        //         printf("Errors: %u while validating for size[uint32_t][%u]\n", errors, sizes[i]);
        //         break;
        //     }
        // }

        // {
        //     printf("`float`: Validation (sequential)\n");
        //     uint32_t errors = validate<float, uint32_t>(sizes[i]);
        //     if(errors > 0) {
        //         printf("Errors: %u while validating for size[float][%u]\n", errors, sizes[i]);
        //         break;
        //     }
        // }

        // {
        //     printf("`float`: Validation (sequential argsort)\n");
        //     uint32_t errors = validate<float, uint32_t>(sizes[i], true, true);
        //     if(errors > 0) {
        //         printf("Errors: %u while validating for size[float][%u]\n", errors, sizes[i]);
        //         break;
        //     }
        // }

        {
            printf("`float`: Validation (random)\n");
            uint32_t errors = validate<float, uint32_t>(sizes[i], false);
            if(errors > 0) {
                printf("Errors: %u while validating for size[float][%u]\n", errors, sizes[i]);
                break;
            }
        }

        // {
        //     printf("`float`: Validation (random argsort)\n");
        //     uint32_t errors = validate<float, uint32_t>(sizes[i], false, true);
        //     if(errors > 0) {
        //         printf("Errors: %u while validating for size[float][%u]\n", errors, sizes[i]);
        //         break;
        //     }
        // }

        // {
        //     printf("`uint8_t`: Validation (random)\n");
        //     uint32_t errors = validate<uint8_t, uint8_t>(sizes[i], false);
        //     if(errors > 0) {
        //         printf("Errors: %u while validating for size[uint8_t][%u]\n", errors, sizes[i]);
        //         break;
        //     }
        // }

        // {
        //     printf("`uint8_t`: Validation (random argsort)\n");
        //     uint32_t errors = validate<uint8_t, uint8_t>(sizes[i], false, true);
        //     if(errors > 0) {
        //         printf("Errors: %u while validating for size[uint8_t][%u]\n", errors, sizes[i]);
        //         break;
        //     }
        // }

        // {
        //     printf("`float16`: Validation (random)\n");
        //     uint32_t errors = validate<__fp16, ushort>(sizes[i], false);
        //     if(errors > 0) {
        //         printf("Errors: %u while validating for size[float16][%u]\n", errors, sizes[i]);
        //         break;
        //     }
        // }

        // {
        //     printf("`float16`: Validation (random argsort)\n");
        //     uint32_t errors = validate<__fp16, ushort>(sizes[i], false, true);
        //     if(errors > 0) {
        //         printf("Errors: %u while validating for size[float16][%u]\n", errors, sizes[i]);
        //         break;
        //     }
        // }

        // {
        //     printf("`bfloat16`: Validation (random)\n");
        //     uint32_t errors = validate<__bf16, ushort>(sizes[i], false);
        //     if(errors > 0) {
        //         printf("Errors: %u while validating for size[bfloat16][%u]\n", errors, sizes[i]);
        //         break;
        //     }
        // }

        // {
        //     printf("`bfloat16`: Validation (random argsort)\n");
        //     uint32_t errors = validate<__bf16, ushort>(sizes[i], false, true);
        //     if(errors > 0) {
        //         printf("Errors: %u while validating for size[bfloat16][%u]\n", errors, sizes[i]);
        //         break;
        //     }
        // }

        // {
        //     printf("`int64_t`: Validation (random argsort)\n");
        //     uint32_t errors = validate<int64_t, uint64_t>(sizes[i], false, true);
        //     if(errors > 0) {
        //         printf("Errors: %u while validating for size[int64_t][%u]\n", errors, sizes[i]);
        //         break;
        //     }
        // }
    }
    return 0;
}