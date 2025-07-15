#include <iostream>
#include <fstream>
#include <cstdio>

#include "agile_host.h"
#include "config.h"
#include "agile_buf_shared.h"
#include "../common/cache_impl.h"
#include "../common/table_impl.h"

#define CPU_CACHE_IMPL DisableCPUCache
#define WRITE_TABLE_IMPL DisableShareTable
#define GPU_CACHE_IMPL SimpleGPUCache<CPU_CACHE_IMPL, WRITE_TABLE_IMPL>

#define AGILE_CTRL AgileCtrl<GPU_CACHE_IMPL, CPU_CACHE_IMPL, WRITE_TABLE_IMPL>
#define AGILE_HOST AgileHost<GPU_CACHE_IMPL, CPU_CACHE_IMPL, WRITE_TABLE_IMPL>
#define AGILE_CACHE_HIERARCHY AgileCacheHierarchy<GPU_CACHE_IMPL, CPU_CACHE_IMPL, WRITE_TABLE_IMPL>
#define AGILE_BUF_ARR AgileBufArrayShared<GPU_CACHE_IMPL, CPU_CACHE_IMPL, WRITE_TABLE_IMPL>



__device__ void compute_krnl(bool enable, AGILE_BUF_ARR &bufArr, unsigned int compute_sim, unsigned int compute_itr){
    
    if(enable){
        __syncthreads();
        for(unsigned int i = 0; i < compute_itr; ++i){
            // for(int j = 0; j < bufArr.size; ++j){ // diable this, the compute_itr can tune CTC in a more fine-grained way
                if(threadIdx.x < bufArr.ctrl->buf_size / sizeof(unsigned int)){
                    ((unsigned int *)((bufArr.buf[0]).data))[threadIdx.x] = ((unsigned int *)((bufArr.buf[0]).data))[threadIdx.x] * compute_sim;
                }
            // }
        }
        __syncthreads();
    }
    
    return;
}


__device__ void issue_load(bool enable, AGILE_BUF_ARR &bufArr0, unsigned int offset, AgileLockChain &chain){
    if(enable){
        bufArr0.load(0, offset, chain);
    }
}

__device__ void wait_load(bool enable, AGILE_BUF_ARR &bufArr0){
    if(enable){
        bufArr0.wait();
    }
}


__global__ void ctc_sync_kernel(AGILE_CTRL * ctrl, AgileBuf * buf0, \
    unsigned int buf_per_blk, unsigned int compute_sim, unsigned int compute_itr, \
    unsigned int total_threads, unsigned int iteration, \
    bool enable_load, bool enable_compute)
{
    AgileLockChain chain;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ AGILE_BUF_ARR bufArr0;
    AGILE_BUF_ARR::init(ctrl, &bufArr0, buf0 + blockIdx.x * buf_per_blk, buf_per_blk);
    for(unsigned int i = 0; i < iteration; ++i){
        
        issue_load(enable_load, bufArr0, i * blockDim.x * buf_per_blk + \
                                                blockIdx.x * buf_per_blk, chain);
        wait_load(enable_load, bufArr0);
        compute_krnl(enable_compute, bufArr0, compute_sim, compute_itr); 
    }
}


__global__ void ctc_async_kernel(AGILE_CTRL * ctrl, AgileBuf * buf0, AgileBuf * buf1, unsigned int buf_per_blk, \
    unsigned int compute_sim, unsigned int compute_itr, unsigned int total_threads, unsigned int iteration,
    bool enable_load, bool enable_compute)
{
    AgileLockChain chain;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ AGILE_BUF_ARR bufArr0;
    AGILE_BUF_ARR::init(ctrl, &bufArr0, buf0 + blockIdx.x * buf_per_blk, buf_per_blk);
    __shared__ AGILE_BUF_ARR bufArr1;
    AGILE_BUF_ARR::init(ctrl, &bufArr1, buf1 + blockIdx.x * buf_per_blk, buf_per_blk);
    for(unsigned int i = 0; i < iteration + 1; ++i){ 
        if(i % 2 == 0){
            wait_load(enable_load && i > 0, bufArr1); 
            issue_load(enable_load && i < iteration, bufArr0, i * blockDim.x * buf_per_blk + blockIdx.x * buf_per_blk, chain);
            compute_krnl(enable_compute && i > 0, bufArr1, compute_sim, compute_itr); 
        }else{
            wait_load(enable_load && i > 0, bufArr0);
            issue_load(enable_load && i < iteration, bufArr1, i * blockDim.x * buf_per_blk + blockIdx.x * buf_per_blk, chain);
            compute_krnl(enable_compute && i > 0, bufArr0, compute_sim, compute_itr);
        }
    }
}

// ./scripts/run.sh bin/ctc-block -itr 10000 -citr 100000 -bpblk 256 -qn 256 -td 1024

__global__ void resetCache(AGILE_CTRL * ctrl, AgileBuf * buf, unsigned int buf_per_blk, unsigned int total_threads){
    unsigned int AGILE_BID = blockIdx.x;
    AgileLockChain chain;
    unsigned int tid = AGILE_BID * blockDim.x + threadIdx.x;
    
    if(tid < buf_per_blk){
        buf[AGILE_BID * buf_per_blk + tid].resetStatus();
    }

    for(unsigned int i = tid; i < ctrl->cache_hierarchy->gpu_cache->slot_num; i += total_threads){
        ctrl->cache_hierarchy->gpu_cache->cache_status[i] = AGILE_GPUCACHE_EMPTY;
        static_cast<SimpleGPUCache<CPU_CACHE_IMPL, WRITE_TABLE_IMPL> *>(ctrl->cache_hierarchy->gpu_cache)->tag_blk_id[i] = -1;
        static_cast<SimpleGPUCache<CPU_CACHE_IMPL, WRITE_TABLE_IMPL> *>(ctrl->cache_hierarchy->gpu_cache)->tag_dev_id[i] = -1;
    }
}

int main(int argc, char ** argv){
    Configs cfg(argc, argv);
    if(cfg.iteration <= 1){
        std::cerr << "Iteration should be greater than 1" << std::endl;
        return 1;
    }
    AGILE_HOST host(0, cfg.slot_size);    

    CPU_CACHE_IMPL c_cache(0, cfg.slot_size); // Disable CPU cache
    WRITE_TABLE_IMPL w_table(0); // Disable write table
    GPU_CACHE_IMPL g_cache(cfg.gpu_slot_num, cfg.slot_size);

    host.setGPUCache(g_cache);
    host.setCPUCache(c_cache);
    host.setShareTable(w_table);

    host.addNvmeDev(cfg.nvme_bar, cfg.bar_size, cfg.ssd_blk_offset, cfg.queue_num, cfg.queue_depth);
    host.initNvme();
    
    AgileBuf * buf0, * buf1, * buf2;
    printf("block dim: %d buf per blk: %d\n", cfg.block_dim, cfg.buf_per_blk);
    host.allocateBuffer(buf0, cfg.block_dim * cfg.buf_per_blk);
    host.allocateBuffer(buf1, cfg.block_dim * cfg.buf_per_blk);
    host.allocateBuffer(buf2, cfg.block_dim * cfg.buf_per_blk);

    
    host.configParallelism(cfg.block_dim, cfg.thread_dim, cfg.agile_dim);
    host.initializeAgile();
   
    
    auto *ctrl = host.getAgileCtrlDevicePtr();
    std::chrono::high_resolution_clock::time_point start0, end0, start1, end1;

    int numBlocksPerSM1 = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM1,
        start_agile_cq_service<GPU_CACHE_IMPL, CPU_CACHE_IMPL, WRITE_TABLE_IMPL>,
        128, // threads per block
        0    // dynamic shared memory
    );

    int numBlocksPerSM2 = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM2,
        ctc_async_kernel,
        cfg.thread_dim, // threads per block
        0    // dynamic shared memory
    );

    int numBlocksPerSM3 = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM3,
        ctc_sync_kernel,
        cfg.thread_dim, // threads per block
        0    // dynamic shared memory
    );

    int numBlocksPerSM4 = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM4,
        resetCache,
        1024, // threads per block
        0    // dynamic shared memory
    );

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int numSMs = prop.multiProcessorCount;

    std::cout << "numBlocksPerSM1: " << numBlocksPerSM1 << " numBlocksPerSM2: " << numBlocksPerSM2 << " numBlocksPerSM3: " << numBlocksPerSM3 << " numSMs: " << numSMs << std::endl;
    std::cout << "block dim: " << cfg.block_dim << " thread dim: " << cfg.thread_dim << std::endl;
    host.startAgile();


    start1 = std::chrono::high_resolution_clock::now();
    host.runKernel(ctc_async_kernel, ctrl, buf1, buf2, cfg.buf_per_blk, cfg.compute_sim, cfg.compute_itr, cfg.block_dim * cfg.thread_dim, cfg.iteration, cfg.enable_load, cfg.enable_compute);
    end1 = std::chrono::high_resolution_clock::now();

    std::cout << "reset cache\n";
    host.runKernel(resetCache, ctrl, buf0, cfg.buf_per_blk, cfg.block_dim * cfg.thread_dim);

    start0 = std::chrono::high_resolution_clock::now();
    host.runKernel(ctc_sync_kernel, ctrl, buf0, cfg.buf_per_blk, cfg.compute_sim, cfg.compute_itr, cfg.block_dim * cfg.thread_dim, cfg.iteration, cfg.enable_load, cfg.enable_compute);
    end0 = std::chrono::high_resolution_clock::now();
    std::cout << "kernel finish\n";
    host.stopAgile();

    std::cout << "finished\n";

    std::cout << "Sync time: " << std::chrono::duration_cast<std::chrono::microseconds>(end0 - start0).count() << " us" << std::endl;
    std::cout << "Async time: " << std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count() << " us" << std::endl;
    std::cout << "Speedup: " << (double)std::chrono::duration_cast<std::chrono::microseconds>(end0 - start0).count() / (double)std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count() << std::endl;

    host.freeBuffer(buf0, cfg.block_dim * cfg.buf_per_blk);
    host.freeBuffer(buf1, cfg.block_dim * cfg.buf_per_blk);
    host.freeBuffer(buf2, cfg.block_dim * cfg.buf_per_blk);
    host.closeNvme();

    return 0;
}