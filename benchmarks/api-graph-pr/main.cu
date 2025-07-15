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

__global__ void pragerank_itr(AGILE_CTRL * ctrl, unsigned int * offsets, unsigned int weight_offset, float damping, float * vec, float * output_vec, float * norm2, unsigned int nodes, unsigned int threads_num){
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= nodes){
        return;
    }
    AgileLockChain chain;
    output_vec[tid] = 0;
    auto agileArr = ctrl->getArrayWrap<unsigned int>(chain);
    unsigned int start = offsets[tid];
    unsigned int end = offsets[tid + 1];
    for(int j = start; j < end; ++j){
        unsigned int col = agileArr[0][j];
        unsigned int val = agileArr[0][weight_offset + j];
        output_vec[tid] += __uint_as_float(val) * vec[col];
    }
    output_vec[tid] *= damping; 
    output_vec[tid] += (1.0 - damping) / nodes;
    float diff = output_vec[tid] - vec[tid];
    atomicAdd(norm2,  diff * diff);
}

int main(int argc, char ** argv){
    Configs cfg(argc, argv);

    AGILE_HOST host(0, cfg.slot_size);    

    CPU_CACHE_IMPL c_cache(0, cfg.slot_size); // Disable CPU cache
    SHARE_TABLE_IMPL w_table(cfg.gpu_slot_num / 4); 
    GPU_CACHE_IMPL g_cache(cfg.gpu_slot_num, cfg.slot_size, cfg.ssd_block_num); // 

    host.setGPUCache(g_cache);
    host.setCPUCache(c_cache);
    host.setShareTable(w_table);

    host.addNvmeDev(cfg.nvme_bar, cfg.bar_size, cfg.ssd_blk_offset, cfg.queue_num, cfg.queue_depth);
    host.initNvme();

    unsigned int * d_offsets, * h_offsets;
    cuda_err_chk(cudaMalloc(&d_offsets, (cfg.node_num + 1) * sizeof(unsigned int)));
    h_offsets = (unsigned int *)malloc((cfg.node_num + 1) * sizeof(unsigned int));
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
    

    cuda_err_chk(cudaMemcpy(d_offsets, h_offsets, (cfg.node_num + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice));

    uint64_t numblocks, numthreads, vertex_count;
    vertex_count = cfg.node_num;
    numthreads = cfg.thread_dim;
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
        pragerank_itr,
        1024, // threads per block
        0    // dynamic shared memory
    );

    host.initializeAgile();

    float * d_vec, * output_vec, * norm2, h_norm2;
    cuda_err_chk(cudaMalloc(&d_vec, cfg.node_num * sizeof(float)));
    cuda_err_chk(cudaMalloc(&output_vec, cfg.node_num * sizeof(float)));
    cuda_err_chk(cudaMalloc(&norm2, sizeof(float)));

    float * h_vec = (float *)malloc(cfg.node_num * sizeof(float));
    float initial_val = 1.0f / cfg.node_num;
    for(unsigned int i = 0; i < cfg.node_num; ++i){
        h_vec[i] = initial_val;
    }
    cuda_err_chk(cudaMemcpy(d_vec, h_vec, cfg.node_num * sizeof(float), cudaMemcpyHostToDevice));
    
    auto *ctrl = host.getAgileCtrlDevicePtr();
    std::chrono::high_resolution_clock::time_point start0, end0, s0, e0;

    host.startAgile();
    start0 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < cfg.iteration; ++i){ // cfg.iteration
        
        h_norm2 = 0;
        cuda_err_chk(cudaMemcpy(norm2, &h_norm2, sizeof(float), cudaMemcpyHostToDevice));
        s0 = std::chrono::high_resolution_clock::now();
        host.runKernel(pragerank_itr, ctrl, d_offsets, cfg.weight_offset, cfg.damping, d_vec, output_vec, norm2, cfg.node_num, cfg.block_dim * cfg.thread_dim);
        e0 = std::chrono::high_resolution_clock::now();
        cuda_err_chk(cudaMemcpy(d_vec, output_vec, sizeof(float) * (cfg.node_num), cudaMemcpyDeviceToDevice));
        cuda_err_chk(cudaMemcpy(&h_norm2, norm2, sizeof(float), cudaMemcpyDeviceToHost));
        h_norm2 = sqrt(h_norm2);
        double itr_time = std::chrono::duration_cast<std::chrono::nanoseconds>(e0 - s0).count();
        std::cout << "itr: " << i << " norm: " << h_norm2 << " itr_time: " << itr_time << " ns" << std::endl;
    }
    end0 = std::chrono::high_resolution_clock::now();
    host.stopAgile();

    
    std::chrono::duration<double> time_span0 = std::chrono::duration_cast<std::chrono::duration<double>>(end0 - start0);
    std::cout << "PR time: " << time_span0.count() << " seconds." << std::endl;

    
    cuda_err_chk(cudaMemcpy(h_vec, d_vec, cfg.node_num * sizeof(float), cudaMemcpyDeviceToHost));
    
    remove(cfg.output_file.c_str());
    std::ofstream ofs(cfg.output_file, std::ios::out);
    for(unsigned int i = 0; i < cfg.node_num; i++){
        ofs << h_vec[i] << std::endl;
    }
    ofs.close();
    free(h_vec);
    free(h_offsets);

    host.closeNvme();

    return 0;
}