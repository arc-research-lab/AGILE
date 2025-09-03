#ifndef AGILE_HELPPER

#define AGILE_HELPPER

#include <cuda_runtime_api.h> 
#include <cuda.h>
#include <memory.h>
#include <cuda/atomic>
#include <cuda/semaphore>
#include <cstdint>
#include <limits>
#include <assert.h>

#include "agile_logger.h"

#define ENABLE_LOGGING 1
#define SHOW_LOGGING 0

#define LOCK_DEBUG 0
#define CPU_CACHE_DIRECT_READ 1  // try to avoid page fault when set to 0, not implemented
#define MESSAGE_ASSERT 1

#define REGISTER_NVME 1

#define REPORT_RACE 1

#define DEBUG_NVME 1

#define FAKE_NVME 0

#define cuda_err_chk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#if ENABLE_LOGGING
    __device__ AgileLogger * logger;
    #define LOGGING(code) code;
#else
    #define LOGGING(...)
#endif

#ifndef __CUDACC__
inline void gpuAssert(int code, const char *file, int line, bool abort=true)
{
    if (code != 0)
    {
	fprintf(stdout,"Assert: %i %s %d\n", code, file, line);
	if (abort) exit(1);
    }
}
#else
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
	fprintf(stdout,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	if (abort) exit(1);
    }
}
#endif


#if MESSAGE_ASSERT
    #define GPU_ASSERT(check, msg) if (!(check)) { printf("Assert Failed %s:%d %s\n", __FILE__, __LINE__, msg); assert(check); }
#else 
    #define GPU_ASSERT(check, msg) assert(check)
#endif

#define TID_TYPE unsigned int
#define BID_TYPE unsigned int

#define TID_NONE 0xFFFFFFFF




#define NVME_DEV_IDX_TYPE unsigned int
#define SSDBLK_TYPE unsigned long long int



template<typename T>
class AGILE_Kernel_TEMP {
public:
    __host__ T * getDevicePtr() {
        T * d_ptr;
        cuda_err_chk(cudaMalloc(&d_ptr, sizeof(T)));
        cuda_err_chk(cudaMemcpy(d_ptr, this, sizeof(T), cudaMemcpyHostToDevice));
        return d_ptr;
    }
};

__device__ void busyWait(int cycles) {
    for (int i = 0; i < cycles; i++) {
        asm volatile("");
    }
}

__device__ void* atomicExchangePtr(void** addr, void* new_ptr) {
    return (void*)atomicExch((unsigned long long*)addr, (unsigned long long)new_ptr);
}

__device__ void* atomicCASPtr(void** addr, void* expected, void* desired) {
    return (void*)atomicCAS(
        (unsigned long long*)addr,
        (unsigned long long)expected,
        (unsigned long long)desired
    );
}

__device__ void* atomicLoadPtr(void** addr) {
    return (void*)atomicCAS(
        (unsigned long long*)addr,
        0ULL,    // expected value
        0ULL     // desired value — same as expected → no change
    );
}

#define LOG_CLOCK(val)  asm volatile ( \
    "mov.u64 %0, %%globaltimer;\n" \
    : "=l"(val) : : "memory" \
)
#endif