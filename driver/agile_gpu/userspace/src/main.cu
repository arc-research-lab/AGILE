#include <iostream>
#include <unistd.h>
#include <sys/types.h>
#include <cstdint>
#include <stdint.h> 
#include <fcntl.h>
#include <stdio.h>

#include "agile_gpu_mem.h"

int main0(int argc, char ** argv){


    // Initialize CUDA
    ASSERTDRV(cuInit(0));
    CUdevice dev;
    ASSERTDRV(cuDeviceGet(&dev, 0));

    CUcontext dev_ctx;
    ASSERTDRV(cuDevicePrimaryCtxRetain(&dev_ctx, dev));
    ASSERTDRV(cuCtxSetCurrent(dev_ctx));

    // Allocate GPU memory
    uint64_t allocated_size = 1024l * 1024l * 1024l * 28l;
    printf("Allocating %lu bytes GPU memory\n", allocated_size);
    CUdeviceptr ptr, aligned_ptr;
    ASSERTDRV(cuMemAlloc(&ptr, allocated_size));
    ASSERTDRV(cuPointerSetAttribute(&ptr, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, ptr));
    aligned_ptr = PAGE_ROUND_UP(ptr, GPU_PAGE_SIZE);
    printf("Allocated GPU memory: %lu bytes at %lx, aligned to %lx\n", allocated_size, ptr, aligned_ptr);

    int fd = open("/dev/AGILE-gpu", O_RDWR);
    if(fd < 0){
        perror("open");
        return -1;
    }


    struct pin_buffer_params params;
    params.vaddr = aligned_ptr;
    params.size = allocated_size - (aligned_ptr - ptr);
    params.p2p_token = 0;
    params.va_space = 0;
    // params.handle = 0;
    // params.table_size = 0;
    // params.table = NULL;

    if(ioctl(fd, IOCTL_PIN_GPU_BUFFER, &params) < 0){
        perror("ioctl");
        close(fd);
        ASSERTDRV(cuMemFree(ptr));
        return -1;
    }


    close(fd);
    ASSERTDRV(cuMemFree(ptr));
    return 0;
}

__global__
void kernel_test(uint32_t * data){
    if(threadIdx.x == 0){
        printf("Hello from GPU kernel! %d\n", data[0]);
        data[0] = 1234;
    }
}

int main(int argc, char ** argv){

    // Initialize CUDA
    ASSERTDRV(cuInit(0));
    CUdevice dev;
    ASSERTDRV(cuDeviceGet(&dev, 0));

    CUcontext dev_ctx;
    ASSERTDRV(cuDevicePrimaryCtxRetain(&dev_ctx, dev));
    ASSERTDRV(cuCtxSetCurrent(dev_ctx));

    AgileGpuMemAllocator allocator;

    AgileGpuMem *mem = allocator.allocateBuf(1024l * 1024l * 1024l * 2l);
    std::cout << "Before kernel, data[0] = " << ((uint32_t *)mem->h_ptr)[0] << std::endl;
    ((uint32_t *)mem->h_ptr)[0] = 42;

    kernel_test<<<1, 32>>>((uint32_t *)mem->d_ptr);
    cudaDeviceSynchronize();
    std::cout << "After kernel, data[0] = " << ((uint32_t *)mem->h_ptr)[0] << std::endl;

    std::cout << "Allocated GPU memory: " << mem->size << " bytes at " << mem->d_ptr 
              << ", physical address: 0x" << std::hex << mem->phy_addr << std::dec << std::endl;


    allocator.freeBuf(mem);

    return 0;
}