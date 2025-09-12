#include <iostream>
#include <unistd.h>
#include <sys/types.h>
#include <cstdint>
#include <stdint.h> 
#include <fcntl.h>
#include <stdio.h>

#include "agile_gpu_mem.h"

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

    struct cpu_dram_buf buf;
    allocator.allocateDramBuf(&buf, 65536);

    uint32_t *cpu_ptr = (uint32_t *)buf.vaddr_user;
    cpu_ptr[0] = 5678;
    // register the buffer with GPU
    ASSERTDRV(cuMemHostRegister(cpu_ptr, 65536, CU_MEMHOSTREGISTER_PORTABLE)); // CUDA error: CUDA_ERROR_INVALID_VALUE
    void * gpu_ptr;
    ASSERTDRV(cuMemHostGetDevicePointer((CUdeviceptr *)&gpu_ptr, cpu_ptr, 0));
    std::cout << "CPU buffer physical address: 0x" << std::hex << buf.phy_addr << std::dec << std::endl;
    std::cout << "CPU buffer GPU pointer: " << gpu_ptr << std::endl;
    std::cout << "Before kernel, CPU buffer[0] = " << cpu_ptr[0] << std::endl;

    kernel_test<<<1, 32>>>((uint32_t *)mem->d_ptr);
    cudaDeviceSynchronize();
    std::cout << "After kernel, data[0] = " << ((uint32_t *)mem->h_ptr)[0] << std::endl;

    std::cout << "Allocated GPU memory: " << mem->size << " bytes at " << mem->d_ptr 
              << ", physical address: 0x" << std::hex << mem->phy_addr << std::dec << std::endl;

    kernel_test<<<1, 32>>>((uint32_t *)gpu_ptr);
    cudaDeviceSynchronize();
    std::cout << "After kernel, CPU buffer[0] = " << cpu_ptr[0] << std::endl;
    ASSERTDRV(cuMemHostUnregister(cpu_ptr));
    allocator.freeDramBuf(&buf);
    allocator.freeBuf(mem);

    return 0;
}