#include "agile_helper.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

using u64 = uint64_t;
#include "../driver/agile_gpu-linux-6.17.0/common/agile_gpu_krnl.h"

#define AGILE_GPU_DEVICE "/dev/AGILE-gpu"
#define PAGE_ROUND_UP(x, n) (((x) + ((n) - 1)) & ~((n) - 1))

inline void agileDrvAssert(CUresult code, const char *file, int line, bool abort = true)
{
    if (code != CUDA_SUCCESS)
    {
        const char *err_name = nullptr;
        cuGetErrorName(code, &err_name);
        fprintf(stdout, "CUDA error: %s %s %d\n", err_name == nullptr ? "unknown" : err_name, file, line);
        if (abort) exit(1);
    }
}

#define agile_drv_chk(ans) { agileDrvAssert((ans), __FILE__, __LINE__); }

/*
* Each page should be 65536 bytes.
*/
unsigned long allocateGPUPinedMem(int device_idx, unsigned int mem_size, CUdeviceptr &d_ptr, void *& h_ptr){
    CUdevice dev;
    agile_drv_chk(cuInit(0));

    int n_devices = 0;
    agile_drv_chk(cuDeviceGetCount(&n_devices));
    if (device_idx < 0 || device_idx >= n_devices) {
        fprintf(stderr, "invalid GPU device index %d\n", device_idx);
        exit(EXIT_FAILURE);
    }

    agile_drv_chk(cuDeviceGet(&dev, device_idx));
    CUcontext dev_ctx;
    agile_drv_chk(cuDevicePrimaryCtxRetain(&dev_ctx, dev));
    agile_drv_chk(cuCtxSetCurrent(dev_ctx));

    const uint64_t aligned_size = PAGE_ROUND_UP(static_cast<uint64_t>(mem_size), GPU_PAGE_SIZE);
    CUdeviceptr raw_d_ptr;
    agile_drv_chk(cuMemAlloc(&raw_d_ptr, aligned_size + GPU_PAGE_SIZE));

    unsigned int sync_memops = 1;
    agile_drv_chk(cuPointerSetAttribute(&sync_memops, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, raw_d_ptr));

    d_ptr = PAGE_ROUND_UP(raw_d_ptr, GPU_PAGE_SIZE);

    int fd = open(AGILE_GPU_DEVICE, O_RDWR);
    if (fd < 0) {
        perror("open /dev/AGILE-gpu");
        exit(EXIT_FAILURE);
    }

    pin_buffer_params params = {};
    params.vaddr = d_ptr;
    params.size = aligned_size;
    params.p2p_token = 0;
    params.va_space = 0;

    if (ioctl(fd, IOCTL_PIN_GPU_BUFFER, &params) < 0) {
        perror("ioctl(IOCTL_PIN_GPU_BUFFER)");
        close(fd);
        exit(EXIT_FAILURE);
    }

    void *mapped_ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, params.phy_addr);
    if (mapped_ptr == MAP_FAILED) {
        perror("mmap GPU");
        ioctl(fd, IOCTL_UNPIN_GPU_BUFFER, &params);
        close(fd);
        exit(EXIT_FAILURE);
    }

    h_ptr = mapped_ptr;
    close(fd);
    return params.phy_addr;
}

unsigned long allocateCPUPinedMem(int fd, void *& h_ptr, unsigned int mem_size) {
    if (mem_size > (4 * 1024 * 1024)) {
        fprintf(stderr, "Each CPU pinned memory cannot exceed 4MB\n");
        exit(EXIT_FAILURE);
    }

    if (fd == -1) {
        perror("open /dev/agile_buffer failed");
        exit(EXIT_FAILURE);
    }

    h_ptr = mmap(nullptr, mem_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (h_ptr == MAP_FAILED || h_ptr == nullptr) {
        perror("mmap failed");
        exit(EXIT_FAILURE);
    }

    unsigned long phy_addr = ((unsigned long *)h_ptr)[0];
    ((unsigned long *)h_ptr)[0] = 0;
    // printf("addr: %ld\n", phy_addr);
    
    return phy_addr;
}