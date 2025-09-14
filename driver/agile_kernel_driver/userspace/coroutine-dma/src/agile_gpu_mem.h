#pragma once

#include <vector>
#include <iostream>
#include <unistd.h>
#include <sys/types.h>
#include <cstdint>
#include <stdint.h> 
#include <fcntl.h>
#include <sys/mman.h>
#include <stdio.h>

#include "agile_gpu_krnl.h"
#include "agile_helper.h"
typedef uint64_t u64;
#define PAGE_ROUND_UP(x, n)     (((x) + ((n) - 1)) & ~((n) - 1))

class AgileGpuMem {
public:
    AgileGpuMem(pin_buffer_params params) : d_ptr(nullptr), h_ptr(nullptr), size(0), buffer_params(params) {}
    void * d_ptr; 
    void * h_ptr;
    uint64_t phy_addr;
    size_t size; // size after alignment
    pin_buffer_params buffer_params; 
};

class AgileGpuMemAllocator {
public:
    AgileGpuMemAllocator() {
        fd = open("/dev/AGILE-gpu", O_RDWR);
        if (fd < 0) {
            perror("open");
            exit(1);
        }
    }
    
    ~AgileGpuMemAllocator() {
        close(fd);
    }

    AgileGpuMem* allocateBuf(uint64_t allocated_size) {

        CUdeviceptr ptr, aligned_ptr;
        ASSERTDRV(cuMemAlloc(&ptr, allocated_size));
        ASSERTDRV(cuPointerSetAttribute(&ptr, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, ptr));
        aligned_ptr = PAGE_ROUND_UP(ptr, GPU_PAGE_SIZE);
        
        pin_buffer_params params;
        params.vaddr = aligned_ptr;
        params.size = allocated_size - (aligned_ptr - ptr);
        params.p2p_token = 0;
        params.va_space = 0;

        AgileGpuMem *mem = new AgileGpuMem(params);
        mem->d_ptr = (void*)aligned_ptr;
        mem->size = params.size;
        if(ioctl(fd, IOCTL_PIN_GPU_BUFFER, &mem->buffer_params) < 0){
            perror("ioctl");
            delete mem;
            return nullptr;
        }
        mem->phy_addr = mem->buffer_params.phy_addr;
        mem->h_ptr = mmap(nullptr, mem->size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, mem->phy_addr);
        allocated_buffers.push_back(mem);
        return mem;
    }

    void freeBuf(AgileGpuMem* mem) {
        auto it = std::find(allocated_buffers.begin(), allocated_buffers.end(), mem);
        if (it != allocated_buffers.end()) {
            munmap(mem->h_ptr, mem->size);
            // unpin GPU buffer
            if(ioctl(fd, IOCTL_UNPIN_GPU_BUFFER, &mem->buffer_params) < 0){
                perror("ioctl");
            }
            ASSERTDRV(cuMemFree((CUdeviceptr)mem->buffer_params.vaddr));
            allocated_buffers.erase(it);
            delete mem;
        }
    }

private:
    int fd;
    std::vector<AgileGpuMem*> allocated_buffers;
};