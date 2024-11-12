#include <stdint.h>
#include <stdio.h>

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>

using namespace std;

#define RADIX 256 // 8-bit RADIX

// Load the `.metal` kernel definition as a string
static const char* kernel = {
    #include "radix.metal"
};

// Random float helper
static inline float random_float() {
    return (float)rand() / (float)RAND_MAX;
}

template<typename T>
bool createData(uint32_t size, T* d_sort, uint32_t* d_idx, T* h_sort, uint32_t* h_idx, bool seq, bool withId) {
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
            h_sort[i] = static_cast<T>(i);
        } else {
            h_sort[i] = random_range(min, max);
        }
        if (withId)
            h_idx[i] = i;
    }

    // cudaError_t cerr;
    
    // cerr = cudaMemcpy(d_sort, h_sort, sortsize, cudaMemcpyHostToDevice);
    // if (cerr) {
    //     printf("[createData memcpy d_sort] Cuda error: %s\n", cudaGetErrorString(cerr));
    //     return true;
    // }
    // if (withId) {
    //     cerr = cudaMemcpy(d_idx, h_idx, idxsize, cudaMemcpyHostToDevice);
    //     if (cerr) {
    //         printf("[createData memcpy d_idx] Cuda error: %s\n", cudaGetErrorString(cerr));
    //         return true;
    //     }
    // }

    // cerr = cudaDeviceSynchronize();
    // if (cerr) {
    //     printf("[createData deviceSync] Cuda error: %s\n", cudaGetErrorString(cerr));
    //     return true;
    // }

    return false;
}

// Helper functions for bit conversions
template<typename T, typename U>
inline uint32_t toBitsCpu(T val) {
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
        return static_cast<uint32_t>(val);
    }
}

// Calculate resources to run
struct Resources {

    static Resources compute(uint32_t size, uint32_t type_size) {
        Resources resc;

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

    Resources res = Resources::compute(size, sizeof(uint32_t));
    // printf("For size[%u]\n---------------\nnumThreadBlocks: %u\nnumUpsweepThreads: %u\nnumScanThreads: %u\nnumDownsweepThreads: %u\ndownsweepSharedSize: %u\ndownsweepKeysPerThreade: %u\nmaxNumElementsInBlock: %u\n\n", size, res.numThreadBlocks, res.numUpsweepThreads, res.numScanThreads, res.numDownsweepThreads, res.downsweepSharedSize, res.downsweepKeysPerThread, res.numElemInBlock);

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

    // Create a library and lookup the kernel function
    MTL::Library* library    = device->newLibrary( NS::String::string(kernel, NS::UTF8StringEncoding), nullptr, &pError );
    MTL::Function* upsweep   = library->newFunction( NS::String::string("RadixUpsweep", NS::UTF8StringEncoding) );
    MTL::Function* scan      = library->newFunction( NS::String::string("RadixScan", NS::UTF8StringEncoding) );
    MTL::Function* downsweep = library->newFunction( NS::String::string("RadixDownsweep", NS::UTF8StringEncoding) );

    if(!upsweep || !scan || !downsweep) {
        __builtin_printf( "%s", pError->localizedDescription()->utf8String() );
        assert(false);
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