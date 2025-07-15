#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>

#include "agile_host.h"
#include "config.h"
#include "../common/cache_impl.h"
#include "../common/table_impl.h"

#include "runner.cuh"
#include "dlrm.h"

// #define CPU_CACHE_IMPL DisableCPUCache
// #define SHARE_TABLE_IMPL SimpleShareTable
// // #define GPU_CACHE_IMPL SimpleGPUCache<CPU_CACHE_IMPL, SHARE_TABLE_IMPL>
// #define GPU_CACHE_IMPL GPUClockReplacementCache<CPU_CACHE_IMPL, SHARE_TABLE_IMPL>

// #define AGILE_CTRL AgileCtrl<GPU_CACHE_IMPL, CPU_CACHE_IMPL, SHARE_TABLE_IMPL>
// #define AGILE_HOST AgileHost<GPU_CACHE_IMPL, CPU_CACHE_IMPL, SHARE_TABLE_IMPL>


__global__ void syncKernel(){}
template<typename Func>
unsigned int getOccupancy(Func kernel) {
    int numBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM,
        kernel,
        1024, // threads per block
        0    // dynamic shared memory
    );
    return numBlocksPerSM;
}


int main(int argc, char ** argv){
    Configs cfg(argc, argv);

    bool enable_compute = cfg.enable_compute == 1;
    bool enable_load = cfg.enable_load == 1;

    AGILE_HOST host(0, cfg.slot_size);    

    CPU_CACHE_IMPL c_cache(0, cfg.slot_size); // Disable CPU cache
    SHARE_TABLE_IMPL w_table(cfg.gpu_slot_num / 4); 
    GPU_CACHE_IMPL g_cache(cfg.gpu_slot_num, cfg.slot_size, cfg.ssd_block_num); // , cfg.ssd_block_num

    host.setGPUCache(g_cache);
    host.setCPUCache(c_cache);
    host.setShareTable(w_table);

    host.addNvmeDev(cfg.nvme_bar, cfg.bar_size, cfg.ssd_blk_offset, cfg.queue_num, cfg.queue_depth);
    host.initNvme();

    getOccupancy(start_agile_cq_service<GPU_CACHE_IMPL, CPU_CACHE_IMPL, SHARE_TABLE_IMPL>);
    getOccupancy(prefetch_embedding_kernel);
    getOccupancy(mean_embedding_kernel);
    getOccupancy(concate_kernel);
    getOccupancy(relu_kernel);
    getOccupancy(sigmoid_kernel);
    getOccupancy(sum_embedding_kernel);

    host.initializeAgile();

    auto *ctrl = host.getAgileCtrlDevicePtr();

    // Create stream
    cudaStream_t stream_compute;
    cudaStreamCreate(&stream_compute);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream_compute);

    // inputs
    unsigned int * h_input_sparse, * d_input_sparse;
    unsigned int * h_vector_sum, * d_vector_sum;

    h_input_sparse = (unsigned int *)malloc(cfg.iteration * cfg.batch_size * cfg.sparse_input_dim * sizeof(unsigned int));
    loadBinFile(cfg.sparse_input_file, h_input_sparse, cfg.iteration * cfg.batch_size * cfg.sparse_input_dim * sizeof(unsigned int), 0);
    cuda_err_chk(cudaMalloc((void **)&d_input_sparse, cfg.iteration * cfg.batch_size * cfg.sparse_input_dim * sizeof(unsigned int)));
    cuda_err_chk(cudaMemcpy(d_input_sparse, h_input_sparse, cfg.iteration * cfg.batch_size * cfg.sparse_input_dim * sizeof(unsigned int), cudaMemcpyHostToDevice));

    h_vector_sum = (unsigned int *)malloc(cfg.iteration * cfg.batch_size * cfg.embedding_vec_dim * sizeof(unsigned int));
    cuda_err_chk(cudaMalloc((void **)&d_vector_sum, cfg.iteration * cfg.batch_size * cfg.embedding_vec_dim * sizeof(unsigned int)));


    host.startAgile();
    std::chrono::high_resolution_clock::time_point start0, end0;
    start0 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < cfg.iteration; ++i){
        unsigned int * d_sparse_in = d_input_sparse + i * cfg.batch_size * cfg.sparse_input_dim;
        unsigned int * d_vector_sum_off = d_vector_sum + i * cfg.batch_size * cfg.embedding_vec_dim;
        // load embedding
        sum_embedding<128>(enable_load, stream_compute, ctrl, d_sparse_in, d_vector_sum_off, 0, cfg.batch_size, cfg.sparse_input_dim, cfg.embedding_vec_dim);
    }
    cuda_err_chk(cudaStreamSynchronize(stream_compute));
    end0 = std::chrono::high_resolution_clock::now();
    host.stopAgile();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end0 - start0);
        std::cout << "Time taken by function: "
            << duration.count() << " milliseconds" << std::endl;
    std::chrono::duration<double> time_span0 = std::chrono::duration_cast<std::chrono::duration<double>>(end0 - start0);

    unsigned int total_cmd = host.h_logger->issued_read;
    double bandwidth = total_cmd * cfg.slot_size;
    bandwidth /= 1024 * 1024 * 1024;
    bandwidth /= time_span0.count();
    std::cout << "Total time: " << time_span0.count() << " s\n";
    std::cout << "Total cmd: " << total_cmd << "\n";
    std::cout << "Bandwidth: " << bandwidth << " GB/s\n";


    // store the sum of embedding to a bin file
    cuda_err_chk(cudaMemcpy(h_vector_sum, d_vector_sum, cfg.iteration * cfg.batch_size * cfg.embedding_vec_dim * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    std::ofstream ofs(cfg.output_file, std::ios::binary);
    if (!ofs) {
        std::cerr << "Error opening output file: " << cfg.output_file << std::endl;
        return 1;
    }
    for(int i = 0; i < cfg.iteration * cfg.batch_size * cfg.embedding_vec_dim; ++i) {
        ofs.write(reinterpret_cast<const char*>(&h_vector_sum[i]), sizeof(unsigned int));
    }
    ofs.close();

    // Cleanup

    cudaStreamDestroy(stream_compute);



    return 0;
}