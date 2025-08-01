#include <iostream>
#include <fstream>
#include <cstdio>

#include "agile_host.h"
#include "config.h"
#include "cache_impl.h"
#include "table_impl.h"



#define CPU_CACHE_IMPL DisableCPUCache
#define SHARE_TABLE_IMPL SimpleShareTable
// #define GPU_CACHE_IMPL SimpleGPUCache<CPU_CACHE_IMPL, SHARE_TABLE_IMPL>
#define GPU_CACHE_IMPL GPUClockReplacementCache<CPU_CACHE_IMPL, SHARE_TABLE_IMPL>

#define AGILE_CTRL AgileCtrl<GPU_CACHE_IMPL, CPU_CACHE_IMPL, SHARE_TABLE_IMPL>
#define AGILE_HOST AgileHost<GPU_CACHE_IMPL, CPU_CACHE_IMPL, SHARE_TABLE_IMPL>


__global__ void gpu_kernel(AGILE_CTRL * ctrl, unsigned int length){
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    AgileLockChain chain;
    if(tid < length){
        auto agileArr = ctrl->getArrayWrap<unsigned int>(chain);
        printf("Thread %d read: %d\n", tid, agileArr[0][tid]);
    }
}

int main(int argc, char ** argv){
    Configs cfg(argc, argv);

    AGILE_HOST host(0, cfg.slot_size);    

    CPU_CACHE_IMPL c_cache(0, cfg.slot_size); // Disable CPU cache
    SHARE_TABLE_IMPL w_table(cfg.gpu_slot_num / 4); 
    GPU_CACHE_IMPL g_cache(cfg.gpu_slot_num, cfg.slot_size, cfg.ssd_block_num);

    host.setGPUCache(g_cache);
    host.setCPUCache(c_cache);
    host.setShareTable(w_table);

    host.addNvmeDev(cfg.nvme_device, cfg.ssd_blk_offset, cfg.queue_num, cfg.queue_depth);
    host.initNvme();

    host.initializeAgile();

    host.queryOccupancy(gpu_kernel, 1024, 0);

    host.startAgile();

    cudaStream_t krnl_stream;
    cudaStreamCreate(&krnl_stream);

    gpu_kernel<<<1, 1024, 0, krnl_stream>>>(host.getAgileCtrlDevicePtr(), 8);
    
    cuda_err_chk(cudaStreamSynchronize(krnl_stream));
    cuda_err_chk(cudaStreamDestroy(krnl_stream));

    host.stopAgile();
    host.closeNvme();
    return 0;
}