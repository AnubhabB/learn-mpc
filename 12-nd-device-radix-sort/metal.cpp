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

// // Specialization for half float if needed
// template<>
// inline half random_range(half min, half max) {
//     float min_f = __half2float(min);
//     float max_f = __half2float(max);
//     return __float2half(min_f + random_float() * (max_f - min_f));
// }

// // Specialization for half float if needed
// template<>
// inline nv_bfloat16 random_range(nv_bfloat16 min, nv_bfloat16 max) {
//     float min_f = __bfloat162float(min);
//     float max_f = __bfloat162float(max);
//     return __float2bfloat16(min_f + random_float() * (max_f - min_f));
// }

// Helper functions for bit conversions
template<typename T, typename U>
inline U toBitsCpu(T val) {
    if constexpr (std::is_same<T, float>::value) {
        uint32_t fuint;
        memcpy(&fuint, &val, sizeof(float));
        return (fuint & 0x80000000) ? ~fuint : fuint ^ 0x80000000;
    // } else if constexpr (std::is_same<T, half>::value) {
    //     ushort bits = __half_as_ushort(val);
    //     return (bits & 0x8000) ? ~bits : bits ^ 0x8000;  // 0x8000 is sign bit for 16-bit
    // } else if constexpr (std::is_same<T, nv_bfloat16>::value) {
    //     ushort bits = __bfloat16_as_ushort(val);
    //     return (bits & 0x8000) ? ~bits : bits ^ 0x8000;  // 0x8000 is still the sign bit
    // } else if constexpr (std::is_same<T, int64_t>::value) {
    //     return static_cast<U>(val) ^ 0x8000000000000000;
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
            min = -24000.0f;
            max = 24000.0f;
        // } else if constexpr (std::is_same<T, half>::value) {
        //     min = -CUDART_MAX_NORMAL_FP16;
        //     max = CUDART_MAX_NORMAL_FP16;
        // } else if constexpr (std::is_same<T, nv_bfloat16>::value) {
        //     min = __float2bfloat16(-6000000.0f);
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
    uint32_t numPartitions;       // number of threadblocks to run for Upsweep and DownsweepPairs kernel
    uint32_t numElemInPartition;  // number of elements per partition
    uint32_t numScanThreads;      // number of threads to launch the scan kernel

    static Resources compute(MTL::Device* device, uint32_t size, uint32_t type_size) {
        Resources resc;

        uint32_t avlUpsweepSharedMem = ((device->maxThreadgroupMemoryLength() - RADIX * sizeof(uint32_t)) * 3)/ 5;

        resc.numElemInPartition  = min(
            (uint32_t)min((uint32_t)avlUpsweepSharedMem / type_size, size) & ~31,
            (uint32_t)2048
        );

        resc.numPartitions   = (size + resc.numElemInPartition - 1) / resc.numElemInPartition;
        resc.numScanThreads  = (resc.numPartitions + LANE_MASK) & ~LANE_MASK;

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
    // printf("For size[%u]\n---------------\nnumThreadBlocks: %u\nnumUpsweepThreads: %u\nnumScanThreads: %u\nnumDownsweepThreads: %u\ndownsweepSharedSize: %u\ndownsweepKeysPerThreade: %u\nmaxNumElementsInBlock: %u\n\n", size, numThreadBlocks, res.numUpsweepThreads, res.numScanThreads, res.numDownsweepThreads, res.downsweepSharedSize, res.downsweepKeysPerThread, res.numElemInBlock);

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
    if constexpr (std::is_same<T, uint8_t>::value) {
        // upFn = upFn + "uint8_t_uint8_t";
        strcpy(upsweepFn, "RadixUpsweep_uint8_t_uint8_t");
    } else if constexpr (std::is_same<T, uint32_t>::value) {
        strcpy(upsweepFn, "RadixUpsweep_uint32_t_uint32_t");
    } else if constexpr (std::is_same<T, float>::value) {
        strcpy(upsweepFn, "RadixUpsweep_float_uint32_t");
    }

    MTL::Function* _up   = library->newFunction( NS::String::string(upsweepFn, NS::UTF8StringEncoding) );
    MTL::Function* _scan = library->newFunction( NS::String::string("RadixScan", NS::UTF8StringEncoding) );
    MTL::Function* _down = library->newFunction( NS::String::string("RadixDownsweep", NS::UTF8StringEncoding) );

    if(!_up) {
        errors += 1;
        __builtin_printf( "UpsweepKernel: function creation error: %s\n", pError->localizedDescription()->utf8String() );
        __builtin_printf( "UpsweepKernel: function creation error(debug): %s\n", pError->debugDescription()->utf8String() );
        return errors;
    }

    // Declarations
    // Array buffers first
    MTL::Buffer* d_sort       = device->newBuffer(sortSize, MTL::ResourceStorageModeShared);
    MTL::Buffer* d_sortAlt    = device->newBuffer(idxSize, MTL::ResourceStorageModeShared);
    MTL::Buffer* d_globalHist = device->newBuffer(radixSize * numPasses, MTL::ResourceStorageModeShared);
    MTL::Buffer* d_passHist   = device->newBuffer(radixSize * resc.numPartitions, MTL::ResourceStorageModeShared);
    MTL::Buffer* d_idx        = device->newBuffer(idxSize, MTL::ResourceStorageModeShared);
    MTL::Buffer* d_idxAlt     = device->newBuffer(idxSize, MTL::ResourceStorageModeShared);

    // Other stuff
    MTL::Buffer* d_size       = device->newBuffer(&size, sizeof(uint32_t), MTL::ResourceStorageModePrivate);
    MTL::Buffer* radixShift   = device->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    MTL::Buffer* elemsPerPart = device->newBuffer(&resc.numElemInPartition, sizeof(uint32_t), MTL::ResourceStorageModePrivate);
    MTL::Buffer* numParts     = device->newBuffer(&resc.numPartitions, sizeof(uint32_t), MTL::ResourceStorageModePrivate);

    // Create some data
    T* sort_buf       = static_cast<T*>(d_sort->contents());
    uint32_t* idx_buf = static_cast<uint32_t*>(d_idx->contents());

    if (createData<T>(size, sort_buf, idx_buf, dataseq, withId)) {
        errors += 1;
        printf("[Pre-pass -> createData] data creation errored out!\n");
        return errors;
    }

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
    
    MTL::Size upThreadsPerGroup     = MTL::Size::Make(_nUpThreads, 1, 1);
    MTL::Size upGroupsPerGrid       = MTL::Size::Make(resc.numPartitions, 1, 1);

    printf("For size[%u]\n---------------\nnumPartitions: %u\nnumUpsweepThreads: %u\nnumScanThreads: %u\nnumDownsweepThreads: %u\ndownsweepSharedSize: %u\ndownsweepKeysPerThreade: %u\nmaxNumElementsInBlock: %u\n\n", size, resc.numPartitions, _nUpThreads, 0, 0, 0, 0, resc.numElemInPartition);

    uint32_t* shift = static_cast<uint32_t*>(radixShift->contents());
    // for (uint32_t pass = 0; pass < numPasses; ++pass) {
    for (uint32_t pass = 0; pass < 1; ++pass) {
        *shift = pass * 8;
        printf("Pass[%u/ %u] Shift[%u]\n", pass, numPasses - 1, *shift);

        // Create a command buffer and a compute encoder from the buffer
        MTL::CommandBuffer* cmdBuffer = cmdQueue->commandBuffer();
        // The upsweep kernel call
        {
            // Get current data being sorted
            T* sortData = static_cast<T*>(d_sort->contents());
            printf("Sort data now: \n");
            for(uint32_t i=0; i<32; ++i) {
                if constexpr (std::is_same<T, float>::value)
                    printf("[%u %f] ", i, sortData[i]);
                // else if constexpr (std::is_same<T, half>::value)
                //     printf("[%u %f] ", i, __half2float(sortData[i]));
                // else if constexpr (std::is_same<T, nv_bfloat16>::value)
                //     printf("[%u %f] ", i, __bfloat162float(sortData[i]));
                else
                    printf("[%u %u] ", i, sortData[i]);
            }
            for(uint32_t i=size - 1; i > size - 32; --i) {
                if constexpr (std::is_same<T, float>::value)
                    printf("[%u %f] ", i, sortData[i]);
                // else if constexpr (std::is_same<T, half>::value)
                //     printf("[%u %f] ", i, __half2float(sortData[i]));
                // else if constexpr (std::is_same<T, nv_bfloat16>::value)
                //     printf("[%u %f] ", i, __bfloat162float(sortData[i]));
                else
                    printf("[%u %u] ", i, sortData[i]);
            }
            printf("\n");

            MTL::ComputeCommandEncoder* upsweepEncoder = cmdBuffer->computeCommandEncoder();
            upsweepEncoder->setComputePipelineState(_upsweepState);
            upsweepEncoder->setBuffer(d_sort, 0, 0);
            upsweepEncoder->setBuffer(d_globalHist, 0, 1);
            upsweepEncoder->setBuffer(d_passHist, 0, 2);
            upsweepEncoder->setBuffer(d_size, 0, 3);
            upsweepEncoder->setBuffer(radixShift, 0, 4);
            upsweepEncoder->setBuffer(elemsPerPart, 0, 5);

            upsweepEncoder->setThreadgroupMemoryLength(RADIX * sizeof(uint32_t), 0);

            upsweepEncoder->dispatchThreadgroups(upGroupsPerGrid, upThreadsPerGroup);
            upsweepEncoder->endEncoding();

            // // Commit and wait for completion
            cmdBuffer->commit();
            cmdBuffer->waitUntilCompleted();

            uint32_t *cpuGlobHist = (uint32_t*)malloc(radixSize);
            uint32_t *gpuGlobHist = static_cast<uint32_t*>(d_globalHist->contents());
            uint32_t *cpuPassHist = (uint32_t*)malloc(radixSize * resc.numPartitions);
            uint32_t *gpuPassHist = static_cast<uint32_t*>(d_passHist->contents());

            // Initialize all zeroes
            for(int i=0; i<RADIX; ++i) {
                cpuGlobHist[i] = 0;
                uint32_t offset = i * resc.numPartitions;
                for(int j=0; j<resc.numPartitions; ++j) {
                    cpuPassHist[offset + j] = 0;
                }
            }

            // Compute CPU histogram
            for (uint32_t i = 0; i < size; i++) {
                uint32_t bits = toBitsCpu<T, U>(sortData[i]);
                uint32_t digit = (bits >> *shift) & RADIX_MASK;
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
            for(uint32_t i = 0; i<resc.numPartitions; ++i) {
                uint32_t blockstart = i * resc.numElemInPartition;
                uint32_t blockend   = min(blockstart + resc.numElemInPartition, size);

                for(uint32_t j=blockstart; j < blockend; ++j) {
                    uint32_t bits = toBitsCpu<T, U>(sortData[j]);
                    bits = ((bits >> *shift) & RADIX_MASK);
                    uint32_t trgt = bits * resc.numPartitions + i;
                    cpuPassHist[trgt] += 1;
                }
            }

            bool passHistError = false;
            for(uint32_t block=0; block < resc.numPartitions; ++block) {
                uint32_t offset = block * RADIX;

                for(uint32_t digit=0; digit < RADIX; digit++) {
                    uint32_t v_idx = offset + digit;
                    if(cpuPassHist[v_idx] != gpuPassHist[v_idx]) {
                        if(!passHistError)
                            printf("Validating `passHist`\n");
                        errors++;
                        passHistError = true;
                        if(errors < 10)
                            printf("Error @ Block[%u] Digit[%u]: Cpu[%u] Gpu[%u]\n", block, digit, cpuPassHist[v_idx], gpuPassHist[v_idx]);
                    }
                }
            }
            printf("Pass hist validation: %s\n", passHistError ? "FAIL" : "PASS");

            bool globalHistError = false;
            for(uint32_t i=0; i<RADIX; ++i) {
                if(cpuGlobHist[i] != gpuGlobHist[i]) {
                    if(!globalHistError)
                        printf("Validating `globalHist`\n");
                    errors++;
                    globalHistError = true;
                    if(errors < 32)
                        printf("Error @ Digit[%u]: Cpu[%u] Gpu[%u]\n", i, cpuGlobHist[i], gpuGlobHist[i]);
                }
            }
            printf("Global hist validation: %s\n", globalHistError ? "FAIL" : "PASS");
            
            free(cpuGlobHist);
            free(cpuPassHist);
        }

        // Launch RadixScan kernel
        {
            uint32_t* passHistBefore = static_cast<uint32_t*>(d_passHist->contents());

            MTL::ComputeCommandEncoder* scanEncoder = cmdBuffer->computeCommandEncoder();
            scanEncoder->setComputePipelineState(_scanState);
            scanEncoder->setBuffer(d_passHist, 0, 0);
            scanEncoder->setBuffer(numParts, 0, 1);
        }

        // Commit and wait for completion
        // cmdBuffer->commit();
        // cmdBuffer->waitUntilCompleted();

        if(errors > 0) {
            printf("Breaking at pass %u\n", pass);
            break;
        }
    }

    return errors;
}

int main() {
    const uint32_t N = 13;
    uint32_t sizes[N] = { 1024, 1120, 2048, 4096, 4113, 7680, 8192, 9216, 16000, 32000, 64000, 128000, 280000 };
    
    // First, test for UpsweepKernel is good?
    // for(uint32_t i = 0; i < N; i++) {
    for(uint32_t i = 0; i < 1; i++) {
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
        //     printf("`float16`: Validation (random)\n");
        //     uint32_t errors = validate<half, ushort>(sizes[i], false);
        //     if(errors > 0) {
        //         printf("Errors: %u while validating for size[float16][%u]\n", errors, sizes[i]);
        //         break;
        //     }
        // }

        // {
        //     printf("`float16`: Validation (random argsort)\n");
        //     uint32_t errors = validate<half, ushort>(sizes[i], false, true);
        //     if(errors > 0) {
        //         printf("Errors: %u while validating for size[float16][%u]\n", errors, sizes[i]);
        //         break;
        //     }
        // }

        // {
        //     printf("`bfloat16`: Validation (random)\n");
        //     uint32_t errors = validate<nv_bfloat16, ushort>(sizes[i], false);
        //     if(errors > 0) {
        //         printf("Errors: %u while validating for size[bfloat16][%u]\n", errors, sizes[i]);
        //         break;
        //     }
        // }

        // {
        //     printf("`bfloat16`: Validation (random argsort)\n");
        //     uint32_t errors = validate<nv_bfloat16, ushort>(sizes[i], false, true);
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