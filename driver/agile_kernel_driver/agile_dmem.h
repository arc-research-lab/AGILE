#pragma once

#include "agile_helper.h"
#include "gdrapi.h"

#include "common.hpp"

#include <iostream>
#include <iomanip>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

typedef struct {
    CUdeviceptr d_ptr;
    void *h_ptr;
    unsigned long addr;
    uint64_t size;
 } pined_gpu_mem;

typedef struct {
    void * ptr;
    uint64_t size;
} managed_cpu_mem;


size_t get_hugepage_size() {
    long size = sysconf(_SC_PAGESIZE);
    if (size == -1) {
        perror("sysconf(_SC_PAGESIZE) failed");
        exit(EXIT_FAILURE);
    }
    return static_cast<size_t>(size);
}

size_t round_up_to_hugepage(size_t size, size_t hugepage_size) {
    return (size + hugepage_size - 1) & ~(hugepage_size - 1);
}

using namespace gdrcopy::test;
/**
* check if the exposed GPU memory is continuous
*/
bool check_continue(long long int table_size, long long int * table){
    for(int i = 0; i < table_size - 1; ++i){
        if(table[i] + 65536 != table[i + 1]){
            MTLOG_ERROR("GPU DMA memory not continuous", table[i], table[i + 1], "(", table_size, ")");
            assert(("GPU DMA memory not continuous", false));
            return false;
        }
    }
    return true;
}

/*
* Each page should be 65536 bytes
*/
// unsigned long allocateGPUPinedMem(int device_idx, unsigned int mem_size, CUdeviceptr &d_ptr, void *& h_ptr){
unsigned long allocateGPUPinedMem(int device_idx, uint32_t mem_size, pined_gpu_mem & pined_mem){
    gdr_t g = gdr_open_safe();
    gdr_mh_t mh;
    void *map_d_ptr = NULL;
    CUdevice dev;
    gpu_mem_handle_t mhandle;
    pined_mem.size = mem_size;

    // CUdeviceptr d_ptr; // exposed GPU memory pointer
    long long int *table; // exposed GPU memory physical addresses table
    gpu_memalloc_fn_t galloc_fn = gpu_mem_alloc;
    ASSERTDRV(cuInit(0));
    int n_devices = 0;
    ASSERTDRV(cuDeviceGetCount(&n_devices));
    int dev_id = device_idx;
    for(int n=0; n<n_devices; ++n) {
        char dev_name[256];
        int dev_pci_domain_id;
        int dev_pci_bus_id;
        int dev_pci_device_id;
        ASSERTDRV(cuDeviceGet(&dev, n));
        ASSERTDRV(cuDeviceGetName(dev_name, sizeof(dev_name) / sizeof(dev_name[0]), dev));
        ASSERTDRV(cuDeviceGetAttribute(&dev_pci_domain_id, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, dev));
        ASSERTDRV(cuDeviceGetAttribute(&dev_pci_bus_id, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev));
        ASSERTDRV(cuDeviceGetAttribute(&dev_pci_device_id, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, dev));
        std::cout << "GPU id:" << n << "; name: " << dev_name 
            << "; Bus id: "
            << std::hex 
            << std::setfill('0') << std::setw(4) << dev_pci_domain_id
            << ":" << std::setfill('0') << std::setw(2) << dev_pci_bus_id
            << ":" << std::setfill('0') << std::setw(2) << dev_pci_device_id
            << std::dec
            << std::endl;
    }
    std::cout << "selecting device " << dev_id << std::endl;
    ASSERTDRV(cuDeviceGet(&dev, dev_id));
    CUcontext dev_ctx;
    ASSERTDRV(cuDevicePrimaryCtxRetain(&dev_ctx, dev));
    ASSERTDRV(cuCtxSetCurrent(dev_ctx));
    ASSERT_EQ(check_gdr_support(dev), true);

    ASSERTDRV(galloc_fn(&mhandle, mem_size, true, true));
    // d_ptr = mhandle.ptr;
    pined_mem.d_ptr = mhandle.ptr;
    long long int table_size;
    gdr_pin_buffer_table(g, pined_mem.d_ptr, mem_size, 0, 0, &mh, &table_size, &table);

    ASSERT_NEQ(mh, null_mh);
    std::cout << "physical address: " << std::hex << (table)[0] << std::dec << std::endl;
    std::cout << "physical continue: " << check_continue(table_size, table) << std::endl;
    std::cout << "table size: " << table_size << std::endl;

    ASSERT_EQ(gdr_map(g, mh, &map_d_ptr, mem_size), 0);
    gdr_info_t info;
    ASSERT_EQ(gdr_get_info(g, mh, &info), 0);
    int off = info.va - pined_mem.d_ptr;
    pined_mem.h_ptr = (void *)((char *)map_d_ptr + off);
    pined_mem.addr = table[0];
    return table[0];
}

unsigned long allocateCPUPinedMem(int fd_mem, void *& h_ptr, unsigned int mem_size) {
    if (mem_size > (4 * 1024 * 1024)) {
        fprintf(stderr, "Each CPU pinned memory cannot exceed 4MB\n");
        exit(EXIT_FAILURE);
    }

    if (fd_mem == -1) {
        perror("open /dev/agile_buffer failed");
        exit(EXIT_FAILURE);
    }

    h_ptr = mmap(nullptr, mem_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_mem, 0);
    if (h_ptr == MAP_FAILED || h_ptr == nullptr) {
        perror("mmap failed");
        exit(EXIT_FAILURE);
    }

    unsigned long phy_addr = ((unsigned long *)h_ptr)[0];
    ((unsigned long *)h_ptr)[0] = 0;
    
    return phy_addr;
}