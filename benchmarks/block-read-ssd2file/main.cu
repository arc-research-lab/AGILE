#include <iostream>
#include <fstream>
#include <cstdio>

#include "agile_host.h"
#include "config.h"
#include "../common/cache_impl.h"
#include "../common/table_impl.h"

#define CPU_CACHE_IMPL DisableCPUCache
#define SHARE_TABLE_IMPL DisableShareTable
#define GPU_CACHE_IMPL SimpleGPUCache<CPU_CACHE_IMPL, SHARE_TABLE_IMPL>

#define AGILE_CTRL AgileCtrl<GPU_CACHE_IMPL, CPU_CACHE_IMPL, SHARE_TABLE_IMPL>
#define AGILE_HOST AgileHost<GPU_CACHE_IMPL, CPU_CACHE_IMPL, SHARE_TABLE_IMPL>

__global__ void read_kernel(AGILE_CTRL * ctrl, AgileBuf * buf, unsigned int block_offset){
    AgileLockChain chain;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    AgileBufPtr bufPtr(buf[tid]);
    bufPtr.resetStatus();
    ctrl->asyncRead(0, block_offset + tid, bufPtr, chain);
    bufPtr.wait();
}

int main(int argc, char ** argv){
    Configs cfg(argc, argv);

    AGILE_HOST host(0, cfg.slot_size);    

    CPU_CACHE_IMPL c_cache(0, cfg.slot_size); // Disable CPU cache
    SHARE_TABLE_IMPL w_table(0); // Disable write table
    GPU_CACHE_IMPL g_cache(cfg.gpu_slot_num, cfg.slot_size);

    host.setGPUCache(g_cache);
    host.setCPUCache(c_cache);
    host.setShareTable(w_table);

    host.addNvmeDev(cfg.nvme_bar, cfg.bar_size, cfg.ssd_blk_offset, cfg.queue_num, cfg.queue_depth);
    host.initNvme();

    int numBlocksPerSM1 = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM1,
        start_agile_cq_service<GPU_CACHE_IMPL, CPU_CACHE_IMPL, SHARE_TABLE_IMPL>,
        1024, // threads per block
        0    // dynamic shared memory
    );

    int numBlocksPerSM2 = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM2,
        read_kernel,
        cfg.thread_dim, // threads per block
        0    // dynamic shared memory
    );

    std::cout << "numBlocksPerSM1: " << numBlocksPerSM1 << std::endl;
    std::cout << "numBlocksPerSM2: " << numBlocksPerSM2 << std::endl;

    AgileBuf * buf;
    host.allocateBuffer(buf, cfg.block_dim * cfg.thread_dim);
    host.configParallelism(cfg.block_dim, cfg.thread_dim, cfg.agile_dim);
    host.initializeAgile();
    remove(cfg.output_file.c_str());
    auto *ctrl = host.getAgileCtrlDevicePtr();
    host.startAgile();
    for(unsigned long i = 0; i < cfg.ssd_blk_num; i += cfg.block_dim * cfg.thread_dim){
        std::cout << "\033[F" << "progress: " << i <<  " / " << cfg.ssd_blk_num << "    " << std::endl;
        host.runKernel(read_kernel, ctrl, buf, i);
        host.appendBuf2File(cfg.output_file, buf, cfg.block_dim * cfg.thread_dim);
    }
    host.stopAgile();
    host.freeBuffer(buf, cfg.block_dim * cfg.thread_dim);
    host.closeNvme();

    return 0;
}