#include <iostream>
#include <fstream>
#include <cstdio>

#include "agile_host.h"
#include "config.h"
#include "../common/cache_impl.h"
#include "../common/table_impl.h"

#define CPU_CACHE_IMPL DisableCPUCache
#define SHARE_TABLE_IMPL SimpleShareTable
// #define GPU_CACHE_IMPL SimpleGPUCache<CPU_CACHE_IMPL, SHARE_TABLE_IMPL>
#define GPU_CACHE_IMPL GPUClockReplacementCache<CPU_CACHE_IMPL, SHARE_TABLE_IMPL>

#define AGILE_CTRL AgileCtrl<GPU_CACHE_IMPL, CPU_CACHE_IMPL, SHARE_TABLE_IMPL>
#define AGILE_HOST AgileHost<GPU_CACHE_IMPL, CPU_CACHE_IMPL, SHARE_TABLE_IMPL>

__global__ void seq_read_kernel(AGILE_CTRL * ctrl){
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long blk_start_idx = tid;
    blk_start_idx *= ctrl->buf_size;
    blk_start_idx /= sizeof(unsigned int);
    AgileLockChain chain;
    auto agileArr = ctrl->getArrayWrap<unsigned int>(chain);
    agileArr[0][blk_start_idx];
}

__global__ void rand_read_kernel(AGILE_CTRL * ctrl, unsigned long * offsets, unsigned int dev_num, unsigned int dev_off, unsigned int req_pre_dev)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= req_pre_dev){
        return;
    }
    AgileLockChain chain;
    unsigned int req_num = dev_num * req_pre_dev;
    auto agileArr = ctrl->getArrayWrap<unsigned int>(chain);
    unsigned long blk_start_idx = offsets[tid];
    blk_start_idx *= ctrl->buf_size;
    blk_start_idx /= sizeof(unsigned int);

    // allow overlap between devices
    for(unsigned int i = dev_off; i < dev_num + dev_off; ++i){
        agileArr[i][blk_start_idx];
    }
}

// __global__ void seq_write_kernel(AGILE_CTRL * ctrl, AgileBufPtr *bufPtr, unsigned int page_count, unsigned long block_offset){

//     AgileLockChain chain;
//     unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

//     for(unsigned int i = tid; i < page_count; i += blockDim.x * gridDim.x){
//         ctrl->writeThroughNvme(0, block_offset + i, bufPtr, chain);
//     }

// }

// __global__ void rand_write_kernel(AGILE_CTRL * ctrl, AgileBufPtr *bufPtr, unsigned long * offsets, unsigned int page_count, unsigned long block_offset){

//     AgileLockChain chain;
//     unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     AgileBufPtr bufPtr(buf[tid]);

//     for(unsigned int i = tid; i < page_count; i += blockDim.x * gridDim.x){
//         unsigned long off = offsets[i];
//         ctrl->writeThroughNvme(0, block_offset + off, bufPtr, chain);
//     }

// }


int main(int argc, char ** argv){
    Configs cfg(argc, argv);

    AGILE_HOST host(0, cfg.slot_size);    

    CPU_CACHE_IMPL c_cache(0, cfg.slot_size); // Disable CPU cache
    SHARE_TABLE_IMPL w_table(cfg.gpu_slot_num / 4); 
    GPU_CACHE_IMPL g_cache(cfg.gpu_slot_num, cfg.slot_size, cfg.ssd_block_num); // , cfg.ssd_block_num

    host.setGPUCache(g_cache);
    host.setCPUCache(c_cache);
    host.setShareTable(w_table);

    // GPU 17:00.0 numa node 0
    
    host.addNvmeDev("0x95400000", 16384, 0, 16, cfg.queue_depth); // 4b:00.0  0
    // host.addNvmeDev("0x96c00000", 16384, 0, 16, cfg.queue_depth); // 98:00.0  1
    host.addNvmeDev("0x97000000", 32768, 0, 64, cfg.queue_depth); // b1:00.0  1
    
    
    host.initNvme();
  
    unsigned int thread_blocks = cfg.req_pre_dev / cfg.thread_dim + (cfg.req_pre_dev % cfg.thread_dim != 0);
    host.configParallelism(thread_blocks, cfg.thread_dim, 1);


    int numBlocksPerSM3 = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM3,
        start_agile_cq_service<GPU_CACHE_IMPL, CPU_CACHE_IMPL, SHARE_TABLE_IMPL>,
        1024, // threads per block
        0    // dynamic shared memory
    );

    int numBlocksPerSM4 = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM4,
        rand_read_kernel,
        1024, // threads per block
        0    // dynamic shared memory
    );

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM4,
        seq_read_kernel,
        1024, // threads per block
        0    // dynamic shared memory
    );

    unsigned long * h_random_offsets, * d_random_offsets;
    h_random_offsets = (unsigned long *)malloc(cfg.req_pre_dev * sizeof(unsigned long));
    for(unsigned int i = 0; i < cfg.req_pre_dev; i++){
        h_random_offsets[i] = ((unsigned int)rand()) % cfg.ssd_block_num;
    }
    cuda_err_chk(cudaMalloc(&d_random_offsets, cfg.req_pre_dev * sizeof(unsigned long)));
    cuda_err_chk(cudaMemcpy(d_random_offsets, h_random_offsets, cfg.req_pre_dev * sizeof(unsigned long), cudaMemcpyHostToDevice));

    host.initializeAgile();

    

    
    auto *ctrl = host.getAgileCtrlDevicePtr();
    std::chrono::high_resolution_clock::time_point start0, end0, s0, e0;

    cudaStream_t cuda_stream;
    cudaStreamCreate(&cuda_stream); 
    

    host.startAgile();
    start0 = std::chrono::high_resolution_clock::now();

    // host.runKernel(seq_read_kernel, ctrl);
    rand_read_kernel<<<thread_blocks, cfg.thread_dim, 0, cuda_stream>>>(ctrl, d_random_offsets, cfg.ssd_num, cfg.ssd_off, cfg.req_pre_dev);
    cuda_err_chk(cudaStreamSynchronize(cuda_stream));

    // host.runKernel(rand_read_kernel, ctrl, d_random_offsets, cfg.ssd_num, cfg.ssd_off);
   
    end0 = std::chrono::high_resolution_clock::now();
    host.stopAgile();

    
    std::chrono::duration<double> time_span0 = std::chrono::duration_cast<std::chrono::duration<double>>(end0 - start0);
    std::cout << "runtime: " << time_span0.count() << " seconds." << std::endl;

    unsigned int total_cmd = host.h_logger->issued_read;
    double bandwidth = total_cmd;
    bandwidth *= cfg.slot_size;
    bandwidth /= 1024 * 1024 * 1024;
    bandwidth /= time_span0.count();
    std::cout << "Total time: " << time_span0.count() << " s\n";
    std::cout << "Total cmd: " << total_cmd << "\n";
    std::cout << "Bandwidth: " << bandwidth << " GB/s\n";
    

    host.closeNvme();

    return 0;
}