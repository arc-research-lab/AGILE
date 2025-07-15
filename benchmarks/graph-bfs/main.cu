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

__global__ void bfs_kernel(AGILE_CTRL * ctrl, \
    unsigned int node_num, unsigned int level, unsigned int * changed, unsigned int * offsets, unsigned int * node_levels, \
    unsigned int thread_num)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    AgileLockChain chain;
        if(tid >= node_num){
            return;
        }

        if(node_levels[tid] == level){
            auto agileArr = ctrl->getArrayWrap<unsigned int>(chain);
            for(unsigned int j = offsets[tid]; j < offsets[tid + 1]; j++){
                unsigned int neighbor = agileArr[0][j];
                if(node_levels[neighbor] == -1){
                    node_levels[neighbor] = level + 1;
                    *changed = 1;
                }
            }
        }
}

int main(int argc, char ** argv){
    Configs cfg(argc, argv);

    AGILE_HOST host(0, cfg.slot_size);    

    CPU_CACHE_IMPL c_cache(0, cfg.slot_size); // Disable CPU cache
    SHARE_TABLE_IMPL w_table(cfg.gpu_slot_num / 4); 
    GPU_CACHE_IMPL g_cache(cfg.gpu_slot_num, cfg.slot_size, cfg.ssd_block_num); // , cfg.ssd_block_num

    host.setGPUCache(g_cache);
    host.setCPUCache(c_cache);
    host.setShareTable(w_table);

    host.addNvmeDev(cfg.nvme_bar, cfg.bar_size, cfg.ssd_blk_offset, cfg.queue_num, cfg.queue_depth);
    host.initNvme();

    unsigned int * d_changed, changed = 0;
    unsigned int * d_offsets, * d_node_levels, * h_offsets, * h_node_levels;
    cuda_err_chk(cudaMalloc(&d_changed, sizeof(unsigned int)));
    cuda_err_chk(cudaMalloc(&d_offsets, (cfg.node_num + 1) * sizeof(unsigned int)));
    cuda_err_chk(cudaMalloc(&d_node_levels, cfg.node_num * sizeof(unsigned int)));
    h_offsets = (unsigned int *)malloc((cfg.node_num + 1) * sizeof(unsigned int));
    h_node_levels = (unsigned int *)malloc(cfg.node_num * sizeof(unsigned int));
    std::ifstream ifs(cfg.offset_file, std::ios::binary);
    if(!ifs.is_open()){
        std::cerr << "Failed to open file: " << cfg.offset_file << std::endl;
        return -1;
    }
    unsigned long finished = 0;
    for(unsigned int i = 0; i < cfg.node_num + 1; i++){
        ifs.read(reinterpret_cast<char*>(&h_offsets[i]), sizeof(unsigned int));
    }
    ifs.close();
    for(unsigned int i = 0; i < cfg.node_num; i++){
        h_node_levels[i] = -1;
    }
    
    h_node_levels[cfg.start_node] = 0;

    cuda_err_chk(cudaMemcpy(d_offsets, h_offsets, (cfg.node_num + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice));
    cuda_err_chk(cudaMemcpy(d_node_levels, h_node_levels, cfg.node_num * sizeof(unsigned int), cudaMemcpyHostToDevice));


    uint64_t numblocks, numthreads, vertex_count;
    vertex_count = cfg.node_num;
    numthreads = 1024;
    unsigned int blockDim = vertex_count / numthreads + 1;

    host.configParallelism(blockDim, numthreads, cfg.agile_dim);

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
        bfs_kernel,
        1024, // threads per block
        0    // dynamic shared memory
    );

    host.initializeAgile();
    
    auto *ctrl = host.getAgileCtrlDevicePtr();
    std::chrono::high_resolution_clock::time_point start0, end0, s0, e0;

    host.startAgile();
    double total_itr_time = 0;
    start0 = std::chrono::high_resolution_clock::now();
    unsigned int level = 0;
    do{
        
        changed = 0;
        cuda_err_chk(cudaMemcpy(d_changed, &changed, sizeof(unsigned int), cudaMemcpyHostToDevice));
        s0 = std::chrono::high_resolution_clock::now();
        host.runKernel(bfs_kernel, ctrl, cfg.node_num, level, d_changed, d_offsets, d_node_levels, cfg.block_dim * cfg.thread_dim);
        e0 = std::chrono::high_resolution_clock::now();
        double itr_time = std::chrono::duration_cast<std::chrono::duration<double>>(e0 - s0).count();
        total_itr_time += itr_time;
        cuda_err_chk(cudaMemcpy(&changed, d_changed, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        
        std::cout << "level: " << level << " changed: " << changed << " itr_time: " << itr_time << " ns" << std::endl;
        level++;
    } while (changed);
    end0 = std::chrono::high_resolution_clock::now();
    host.stopAgile();

    
    std::chrono::duration<double> time_span0 = std::chrono::duration_cast<std::chrono::duration<double>>(end0 - start0);
    std::cout << "BFS time: " << total_itr_time << " seconds." << std::endl;
    cuda_err_chk(cudaMemcpy(h_node_levels, d_node_levels, cfg.node_num * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    
    remove(cfg.output_file.c_str());
    std::ofstream ofs(cfg.output_file, std::ios::out | std::ios::binary);
    for(unsigned int i = 0; i < cfg.node_num; i++){
        ofs.write(reinterpret_cast<const char*>(&h_node_levels[i]), sizeof(unsigned int));
    }
    ofs.close();

    host.closeNvme();

    return 0;
}