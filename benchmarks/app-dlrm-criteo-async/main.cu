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

    host.initializeAgile();

    auto *ctrl = host.getAgileCtrlDevicePtr();

    // Create stream
    cudaStream_t stream_compute;
    cudaStreamCreate(&stream_compute); // 

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream_compute);

    // inputs
    float * h_input_dense, * d_input_dense;
    unsigned int * h_input_sparse, * d_input_sparse;

    std::chrono::high_resolution_clock::time_point start0, end0;
    std::cout << "iteration: " << cfg.iteration << std::endl;

    std::vector<unsigned int> bottom_layer_config;
    std::vector<unsigned int> top_layer_config;

    std::stringstream bottom_ss(cfg.bottom_layer);
    std::stringstream top_ss(cfg.top_layer);
    std::string token;

    while (std::getline(bottom_ss, token, '-')) {
        bottom_layer_config.push_back(std::stoi(token));
    }

    while (std::getline(top_ss, token, '-')) {
        top_layer_config.push_back(std::stoi(token));
    }
    std::cout << "bottom layer config: ";
    for(auto i : bottom_layer_config){
        std::cout << i << " ";
    }
    std::cout << std::endl;

    std::cout << "top layer config: ";
    for(auto i : top_layer_config){
        std::cout << i << " ";
    }
    std::cout << std::endl;

    float * d_vector_mean;
    float * d_all_features;
    cuda_err_chk(cudaMalloc((void **)&d_vector_mean, cfg.batch_size * cfg.embedding_vec_dim * sizeof(float)));
    cuda_err_chk(cudaMalloc((void **)&d_all_features, cfg.batch_size * 2 * cfg.embedding_vec_dim * sizeof(float)));

    float * h_bottom_projection, * d_bottom_projection;
    float * h_projection_layer, * d_projection_layer;

    float * d_bottom_projection_output;
    float ** h_bottom_layers, ** d_bottom_layers;
    float ** d_bottom_activations;

    float * d_top_in;
    cuda_err_chk(cudaMalloc((void **)&d_top_in, cfg.batch_size * top_layer_config[0] * sizeof(float)));

    float ** h_top_layers, ** d_top_layers;
    float ** d_top_activations;

    float * h_bottom_2_embdim, * d_bottom_2_embdim;
    float * d_projection_activations;
    float * h_output_layer, * d_output_layer;

    h_bottom_layers = (float **)malloc(bottom_layer_config.size() * sizeof(float *));
    h_top_layers = (float **)malloc(top_layer_config.size() * sizeof(float *));

    d_bottom_layers = (float **)malloc(bottom_layer_config.size() * sizeof(float *));
    d_top_layers = (float **)malloc(top_layer_config.size() * sizeof(float *));
    unsigned long offset = 0;

    // bottom projection
    h_bottom_projection = (float *)malloc(cfg.dense_input_dim * bottom_layer_config[0] * sizeof(float));
    loadBinFile(cfg.weights_file, h_bottom_projection, cfg.dense_input_dim * bottom_layer_config[0] * sizeof(float), offset);
    cuda_err_chk(cudaMalloc((void **)&d_bottom_projection, cfg.dense_input_dim * bottom_layer_config[0] * sizeof(float)));
    cuda_err_chk(cudaMemcpy(d_bottom_projection, h_bottom_projection, cfg.dense_input_dim * bottom_layer_config[0] * sizeof(float), cudaMemcpyHostToDevice));
    offset += cfg.dense_input_dim * bottom_layer_config[0] * sizeof(float);

    cuda_err_chk(cudaMalloc((void **)&d_bottom_projection_output, cfg.batch_size * bottom_layer_config[0] * sizeof(float)));

    // bottom layers
    unsigned int prev_size = bottom_layer_config[0];
    for(unsigned int i = 0; i < bottom_layer_config.size(); i++){
        h_bottom_layers[i] = (float *)malloc(prev_size * bottom_layer_config[i] * sizeof(float));
        loadBinFile(cfg.weights_file, h_bottom_layers[i], prev_size * bottom_layer_config[i] * sizeof(float), offset);
        cuda_err_chk(cudaMalloc((void **)&d_bottom_layers[i], prev_size * bottom_layer_config[i] * sizeof(float)));
        cuda_err_chk(cudaMemcpy(d_bottom_layers[i], h_bottom_layers[i], prev_size * bottom_layer_config[i] * sizeof(float), cudaMemcpyHostToDevice));
        offset += prev_size * bottom_layer_config[i] * sizeof(float);
        prev_size = bottom_layer_config[i];
    }

    // bottom activations
    d_bottom_activations = (float **)malloc(bottom_layer_config.size() * sizeof(float *));
    for(unsigned int i = 0; i < bottom_layer_config.size(); i++){
        cuda_err_chk(cudaMalloc((void **)&d_bottom_activations[i], cfg.batch_size * bottom_layer_config[i] * sizeof(float)));
    }

    // project bottom output to embedding dim
    h_bottom_2_embdim = (float *)malloc(prev_size * cfg.embedding_vec_dim * sizeof(float));
    loadBinFile(cfg.weights_file, h_bottom_2_embdim, prev_size * cfg.embedding_vec_dim * sizeof(float), offset);
    cuda_err_chk(cudaMalloc((void **)&d_bottom_2_embdim, prev_size * cfg.embedding_vec_dim * sizeof(float)));
    cuda_err_chk(cudaMemcpy(d_bottom_2_embdim, h_bottom_2_embdim, prev_size * cfg.embedding_vec_dim * sizeof(float), cudaMemcpyHostToDevice));
    offset += prev_size * cfg.embedding_vec_dim * sizeof(float);

    cuda_err_chk(cudaMalloc((void **)&d_projection_activations, cfg.batch_size * cfg.embedding_vec_dim * sizeof(float)));

    // projection all features to top layer
    h_projection_layer = (float *)malloc(cfg.embedding_vec_dim * 2 * top_layer_config[0] * sizeof(float));
    loadBinFile(cfg.weights_file, h_projection_layer, cfg.embedding_vec_dim * 2 * top_layer_config[0] * sizeof(float), offset);
    cuda_err_chk(cudaMalloc((void **)&d_projection_layer, cfg.embedding_vec_dim * 2 * top_layer_config[0] * sizeof(float)));
    cuda_err_chk(cudaMemcpy(d_projection_layer, h_projection_layer, cfg.embedding_vec_dim * 2 * top_layer_config[0] * sizeof(float), cudaMemcpyHostToDevice));
    offset += cfg.embedding_vec_dim * 2 * top_layer_config[0] * sizeof(float);
    cuda_err_chk(cudaMalloc((void **)&d_top_in, cfg.batch_size * top_layer_config[0] * sizeof(float)));


    prev_size = top_layer_config[0];
    // top layers
    for(unsigned int i = 0; i < top_layer_config.size(); i++){
        h_top_layers[i] = (float *)malloc(prev_size * top_layer_config[i] * sizeof(float));
        loadBinFile(cfg.weights_file, h_top_layers[i], prev_size * top_layer_config[i] * sizeof(float), offset);
        cuda_err_chk(cudaMalloc((void **)&d_top_layers[i], prev_size * top_layer_config[i] * sizeof(float)));
        cuda_err_chk(cudaMemcpy(d_top_layers[i], h_top_layers[i], prev_size * top_layer_config[i] * sizeof(float), cudaMemcpyHostToDevice));
        offset += prev_size * top_layer_config[i] * sizeof(float);
        prev_size = top_layer_config[i];
    }

    // top activations
    d_top_activations = (float **)malloc(top_layer_config.size() * sizeof(float *));
    for(unsigned int i = 0; i < top_layer_config.size(); i++){
        cuda_err_chk(cudaMalloc((void **)&d_top_activations[i], cfg.batch_size * top_layer_config[i] * sizeof(float)));
    }

    h_output_layer = (float *)malloc(prev_size * sizeof(float));
    loadBinFile(cfg.weights_file, h_output_layer, prev_size * sizeof(float), offset);
    cuda_err_chk(cudaMalloc((void **)&d_output_layer, prev_size * sizeof(float)));
    cuda_err_chk(cudaMemcpy(d_output_layer, h_output_layer, prev_size * sizeof(float), cudaMemcpyHostToDevice));
    offset += prev_size * sizeof(float);
    prev_size = 1;

    // load input data
    h_input_dense = (float *)malloc(cfg.iteration * cfg.batch_size * cfg.dense_input_dim * sizeof(float));
    h_input_sparse = (unsigned int *)malloc(cfg.iteration * cfg.batch_size * cfg.sparse_input_dim * sizeof(unsigned int));
    loadBinFile(cfg.dense_input_file, h_input_dense, cfg.iteration * cfg.batch_size * cfg.dense_input_dim * sizeof(float), 0);
    loadBinFile(cfg.sparse_input_file, h_input_sparse, cfg.iteration * cfg.batch_size * cfg.sparse_input_dim * sizeof(unsigned int), 0);

    cuda_err_chk(cudaMalloc((void **)&d_input_dense, cfg.iteration * cfg.batch_size * cfg.dense_input_dim * sizeof(float)));
    cuda_err_chk(cudaMalloc((void **)&d_input_sparse, cfg.iteration * cfg.batch_size * cfg.sparse_input_dim * sizeof(unsigned int)));
    cuda_err_chk(cudaMemcpy(d_input_dense, h_input_dense, cfg.iteration * cfg.batch_size * cfg.dense_input_dim * sizeof(float), cudaMemcpyHostToDevice));
    cuda_err_chk(cudaMemcpy(d_input_sparse, h_input_sparse, cfg.iteration * cfg.batch_size * cfg.sparse_input_dim * sizeof(unsigned int), cudaMemcpyHostToDevice));

    // output
    float * h_dlrm_output;
    float * d_dlrm_output;
    h_dlrm_output = (float *)malloc(cfg.iteration * cfg.batch_size * sizeof(float));
    cuda_err_chk(cudaMalloc((void **)&d_dlrm_output, cfg.iteration * cfg.batch_size * sizeof(float)));

    // warm up run
    // projection bottom
    runCublasFP32(enable_compute, handle, cfg.batch_size, cfg.dense_input_dim, bottom_layer_config[0], 1.0f, d_input_dense, d_bottom_projection, 0.0f, d_bottom_projection_output);
    relu(enable_compute, stream_compute, d_bottom_projection_output, cfg.batch_size * bottom_layer_config[0]);
    // bottom layers warm-up
    for (unsigned int i = 0; i < bottom_layer_config.size(); i++) {
        runCublasFP32(enable_compute, handle, cfg.batch_size, 
                        (i == 0 ? bottom_layer_config[0] : bottom_layer_config[i - 1]), 
                        bottom_layer_config[i], 
                        1.0f, 
                        (i == 0 ? d_bottom_projection_output : d_bottom_activations[i - 1]), 
                        d_bottom_layers[i], 
                        0.0f, 
                        d_bottom_activations[i]);
        relu(enable_compute, stream_compute, d_bottom_activations[i], cfg.batch_size * bottom_layer_config[i]);
    }

    // project bottom output to embedding dim
    runCublasFP32(enable_compute, handle, cfg.batch_size, bottom_layer_config.back(), cfg.embedding_vec_dim, 1.0f, d_bottom_activations[bottom_layer_config.size() - 1], d_bottom_2_embdim, 0.0f, d_projection_activations);
    relu(enable_compute, stream_compute, d_projection_activations, cfg.batch_size * cfg.embedding_vec_dim);

    // concatenate the dense and sparse input
    concate<128>(enable_compute, stream_compute, d_projection_activations, d_vector_mean, d_all_features, cfg.batch_size, cfg.embedding_vec_dim);
    
    // projection all features to top layer
    runCublasFP32(enable_compute, handle, cfg.batch_size, cfg.embedding_vec_dim * 2, top_layer_config[0], 1.0f, d_all_features, d_projection_layer, 0.0f, d_top_in);
    relu(enable_compute, stream_compute, d_top_in, cfg.batch_size * top_layer_config[0]);

    // top layers warm-up
    for (unsigned int i = 0; i < top_layer_config.size(); i++) {
        runCublasFP32(enable_compute, handle, cfg.batch_size, 
                        (i == 0 ? top_layer_config[0] : top_layer_config[i - 1]), 
                        top_layer_config[i], 
                        1.0f, 
                        (i == 0 ? d_top_in : d_top_activations[i - 1]), 
                        d_top_layers[i], 
                        0.0f, 
                        d_top_activations[i]);
        relu(enable_compute, stream_compute, d_top_activations[i], cfg.batch_size * top_layer_config[i]);
    }

    // top output
    runCublasFP32(enable_compute, handle, cfg.batch_size, top_layer_config.back(), 1, 1.0f, d_top_activations[top_layer_config.size() - 1], d_output_layer, 0.0f, d_dlrm_output);

    // sigmoid
    sigmoid(enable_compute, stream_compute, d_dlrm_output, cfg.batch_size);
    cuda_err_chk(cudaStreamSynchronize(stream_compute));
    

    host.startAgile();
    start0 = std::chrono::high_resolution_clock::now();
    for(int itr = 0; itr < cfg.iteration + 1; ++itr){
        // fetch for the next iteration
        unsigned int * d_sparse_in0 = d_input_sparse + itr * cfg.batch_size * cfg.sparse_input_dim;
        prefetch_embedding<128>(enable_load && itr < cfg.iteration, stream_compute, ctrl, d_sparse_in0, 0, cfg.batch_size, cfg.sparse_input_dim, cfg.embedding_vec_dim);

        

        float * d_dense_in = d_input_dense + (itr - 1) * cfg.batch_size * cfg.dense_input_dim;
        // projection bottom
        runCublasFP32(enable_compute && itr > 0, handle, cfg.batch_size, cfg.dense_input_dim, bottom_layer_config[0], 1.0f, d_dense_in, d_bottom_projection, 0.0f, d_bottom_projection_output);
        relu(enable_compute && itr > 0, stream_compute, d_bottom_projection_output, cfg.batch_size * bottom_layer_config[0]);

        // bottom layers
        for (unsigned int i = 0; i < bottom_layer_config.size(); i++) {
            runCublasFP32(enable_compute && itr > 0, handle, cfg.batch_size, 
                            (i == 0 ? bottom_layer_config[0] : bottom_layer_config[i - 1]), 
                            bottom_layer_config[i], 
                            1.0f, 
                            (i == 0 ? d_bottom_projection_output : d_bottom_activations[i - 1]), 
                            d_bottom_layers[i], 
                            0.0f, 
                            d_bottom_activations[i]);
            relu(enable_compute && itr > 0, stream_compute, d_bottom_activations[i], cfg.batch_size * bottom_layer_config[i]);
        }

        // project bottom output to embedding dim
        runCublasFP32(enable_compute && itr > 0, handle, cfg.batch_size, bottom_layer_config.back(), cfg.embedding_vec_dim, 1.0f, d_bottom_activations[bottom_layer_config.size() - 1], d_bottom_2_embdim, 0.0f, d_projection_activations);
        relu(enable_compute && itr > 0, stream_compute, d_projection_activations, cfg.batch_size * cfg.embedding_vec_dim);

        // load embedding
        unsigned int * d_sparse_in1 = d_input_sparse + (itr - 1) * cfg.batch_size * cfg.sparse_input_dim;
        mean_embedding<128>(enable_load && itr > 0, stream_compute, ctrl, d_sparse_in1, d_vector_mean, 0, cfg.batch_size, cfg.sparse_input_dim, cfg.embedding_vec_dim);

        // concatenate the dense and sparse input
        concate<128>(enable_compute && itr > 0, stream_compute, d_projection_activations, d_vector_mean, d_all_features, cfg.batch_size, cfg.embedding_vec_dim);

        // projection all features to top layer
        runCublasFP32(enable_compute && itr > 0, handle, cfg.batch_size, cfg.embedding_vec_dim * 2, top_layer_config[0], 1.0f, d_all_features, d_projection_layer, 0.0f, d_top_in);
        relu(enable_compute && itr > 0, stream_compute, d_top_in, cfg.batch_size * top_layer_config[0]);

        // top layers
        for (unsigned int i = 0; i < top_layer_config.size(); i++) {
            runCublasFP32(enable_compute && itr > 0, handle, cfg.batch_size, 
                            (i == 0 ? top_layer_config[0] : top_layer_config[i - 1]), 
                            top_layer_config[i], 
                            1.0f, 
                            (i == 0 ? d_top_in : d_top_activations[i - 1]), 
                            d_top_layers[i], 
                            0.0f, 
                            d_top_activations[i]);
            relu(enable_compute && itr > 0, stream_compute, d_top_activations[i], cfg.batch_size * top_layer_config[i]);
        }

        // top output
        float * d_out = d_dlrm_output + (itr - 1) * cfg.batch_size;
        runCublasFP32(enable_compute && itr > 0, handle, cfg.batch_size, top_layer_config.back(), 1, 1.0f, d_top_activations[top_layer_config.size() - 1], d_output_layer, 0.0f, d_out);

        // sigmoid
        sigmoid(enable_compute && itr > 0, stream_compute, d_out, cfg.batch_size);
    }
    cuda_err_chk(cudaStreamSynchronize(stream_compute));
    end0 = std::chrono::high_resolution_clock::now();
    host.stopAgile();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end0 - start0);
        std::cout << "Time taken by function: "
            << duration.count() << " milliseconds" << std::endl;
    std::chrono::duration<double> time_span0 = std::chrono::duration_cast<std::chrono::duration<double>>(end0 - start0);

    unsigned int total_cmd = host.h_logger->issued_read;
    double bandwidth = total_cmd;
    bandwidth *= cfg.slot_size;
    bandwidth /= 1024 * 1024 * 1024;
    bandwidth /= time_span0.count();
    std::cout << "Total time: " << time_span0.count() << " s\n";
    std::cout << "Total cmd: " << total_cmd << "\n";
    std::cout << "Bandwidth: " << bandwidth << " GB/s\n";

    // Cleanup
    cudaStreamDestroy(stream_compute);

    host.closeNvme();
    return 0;
}