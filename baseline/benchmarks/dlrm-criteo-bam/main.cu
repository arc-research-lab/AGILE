// #include "cfg.h"
#include "runner.cuh"
#include "dlrm.h"

#define __ATOMIC_THREAD 10

#include <cuda_runtime_api.h> 
#include <cooperative_groups.h>
#include <cuda.h>

#ifdef __DIS_CLUSTER__
#include <sisci_api.h>
#endif
#include "config.h"


#define UINT64MAX 0xFFFFFFFFFFFFFFFF
#define MYINFINITY 0xFFFFFFFF
#define WARP_SHIFT 5
#define WARP_SIZE 32
#define CHUNK_SHIFT 3
#define CHUNK_SIZE (1 << CHUNK_SHIFT)
#define BLOCK_NUM 1024ULL

using error = std::runtime_error;
using std::string;

const char* const sam_ctrls_paths[] = {"/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm4", "/dev/libnvm9", "/dev/libnvm2", "/dev/libnvm3", "/dev/libnvm5", "/dev/libnvm6", "/dev/libnvm7", "/dev/libnvm8"};
const char* const intel_ctrls_paths[] = {"/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm2", "/dev/libnvm3", "/dev/libnvm4", "/dev/libnvm5", "/dev/libnvm6", "/dev/libnvm7", "/dev/libnvm8", "/dev/libnvm9"};

int main(int argc, char** argv) {

    Configs cfg(argc, argv);

    cudaDeviceProp properties;
    if (cudaGetDeviceProperties(&properties, cfg.cudaDevice) != cudaSuccess)
    {
        fprintf(stderr, "Failed to get CUDA device properties\n");
        return 1;
    }

    try {

        cuda_err_chk(cudaSetDevice(cfg.cudaDevice));
        std::vector<Controller*> ctrls(cfg.n_ctrls);
        for (size_t i = 0 ; i < cfg.n_ctrls; i++)
            ctrls[i] = new Controller(cfg.ssdtype == 0 ? sam_ctrls_paths[i] : intel_ctrls_paths[i], cfg.nvmNamespace, cfg.cudaDevice, cfg.queue_depth, cfg.queue_num);

        uint32_t page_size = cfg.slot_size;
        uint64_t n_pages = cfg.gpu_slot_num;
        uint64_t total_cache_size = (page_size * n_pages);
        std::cout << "page size: " << page_size << std::endl;
        std::cout << "page num: " << n_pages << std::endl;
        std::cout << "total cache size: " << total_cache_size << std::endl;
        printf("===========================\n");

        page_cache_t h_pc(page_size, n_pages, cfg.cudaDevice, ctrls[0][0], (uint32_t) 64, ctrls);
        
        //QueuePair* d_qp;
        page_cache_t* d_pc = (page_cache_t*) (h_pc.d_pc_ptr);
        #define TYPE uint32_t
        uint64_t n_elems = cfg.numElems; 
        uint64_t t_size = n_elems * sizeof(TYPE);

        range_t<uint32_t> h_range((uint32_t)0, (uint64_t)n_elems, (uint64_t)0, (uint64_t)(t_size/page_size) + 1, (uint64_t)0, (uint64_t)page_size, &h_pc, cfg.cudaDevice);
        range_t<uint32_t>* d_range = (range_t<uint32_t>*) h_range.d_range_ptr;

        std::vector<range_t<uint32_t>*> vr(1);
        vr[0] = & h_range;
        array_t<uint32_t> nvme_mem(n_elems, 0, vr, cfg.cudaDevice);

        std::cout << "finished creating range\n";
        std::cout << "atlaunch kernel\n";
        printf("===========================\n");
        char st[15];
        cuda_err_chk(cudaDeviceGetPCIBusId(st, 15, cfg.cudaDevice));
        std::cout << st << std::endl;


        bool enable_compute = cfg.enable_compute == 1;
        bool enable_load = cfg.enable_load == 1;

        // Create stream
        cudaStream_t stream_compute;
        cudaStreamCreate(&stream_compute); // 

        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSetStream(handle, stream_compute);

        std::cout << std::dec << std::endl;

        std::cout << "cfg.iteration: " << cfg.iteration << std::endl;
        std::cout << "cfg.batch_size: " << cfg.batch_size << std::endl;
        std::cout << "cfg.embedding_vec_dim: " << cfg.embedding_vec_dim << std::endl;
        std::cout << "cfg.dense_input_dim: " << cfg.dense_input_dim << std::endl;
        std::cout << "cfg.sparse_input_dim: " << cfg.sparse_input_dim << std::endl;
        std::cout << "cfg.n_ctrls: " << cfg.n_ctrls << std::endl;
        std::cout << "cfg.queueDepth: " << cfg.queue_depth << std::endl;
        std::cout << "cfg.numQueues: " << cfg.queue_num << std::endl;
        std::cout << "cfg.enable_load: " << cfg.enable_load << std::endl;
        std::cout << "cfg.enable_compute: " << cfg.enable_compute << std::endl;
        std::cout << "cfg.ssdtype: " << cfg.ssdtype << std::endl;


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
        
        // cuda_err_chk(cudaMemcpy(h_input_sparse, d_input_sparse, cfg.iteration * cfg.batch_size * cfg.sparse_input_dim * sizeof(unsigned int), cudaMemcpyDeviceToHost));

        // for(int i = 0; i < cfg.iteration * cfg.batch_size * cfg.sparse_input_dim; ++i){
        //     std::cout << "h_input_sparse[" << i << "]: " << h_input_sparse[i] << " \n";
        // }
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
    
        start0 = std::chrono::high_resolution_clock::now();
        for(int itr = 0; itr < cfg.iteration; ++itr){
            // fetch for the next iteration
            unsigned int * d_sparse_in = d_input_sparse + itr * cfg.batch_size * cfg.sparse_input_dim;
            // load embedding
            mean_embedding<128>(enable_load, stream_compute, nvme_mem.d_array_ptr, d_sparse_in, d_vector_mean, 0, cfg.batch_size, cfg.sparse_input_dim, cfg.embedding_vec_dim);

            float * d_dense_in = d_input_dense + itr * cfg.batch_size * cfg.dense_input_dim;
            // projection bottom
            runCublasFP32(enable_compute, handle, cfg.batch_size, cfg.dense_input_dim, bottom_layer_config[0], 1.0f, d_dense_in, d_bottom_projection, 0.0f, d_bottom_projection_output);
            relu(enable_compute, stream_compute, d_bottom_projection_output, cfg.batch_size * bottom_layer_config[0]);

            // bottom layers
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

            // top layers
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
            float * d_out = d_dlrm_output + itr * cfg.batch_size;
            runCublasFP32(enable_compute, handle, cfg.batch_size, top_layer_config.back(), 1, 1.0f, d_top_activations[top_layer_config.size() - 1], d_output_layer, 0.0f, d_out);

            // sigmoid
            sigmoid(enable_compute, stream_compute, d_out, cfg.batch_size);

            // cuda_err_chk(cudaStreamSynchronize(stream_compute));

            // std::cout << "itr: " << itr << " accum cmd: " << host.h_logger->issued_read << std::endl;
        }

        cuda_err_chk(cudaStreamSynchronize(stream_compute));
        
        end0 = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end0 - start0);
            std::cout << "Time taken by function: "
                << duration.count() << " milliseconds" << std::endl;
        std::chrono::duration<double> time_span0 = std::chrono::duration_cast<std::chrono::duration<double>>(end0 - start0);

        unsigned int total_cmd = nvme_mem.print_reset_stats();
        double bandwidth = total_cmd;
        bandwidth *= cfg.slot_size;
        bandwidth /= 1024 * 1024 * 1024;
        bandwidth /= time_span0.count();
        std::cout << "Total time: " << time_span0.count() << " s\n";
        std::cout << "Total cmd: " << total_cmd << "\n";
        std::cout << "Bandwidth: " << bandwidth << " GB/s\n";
        


        for (size_t i = 0 ; i < cfg.n_ctrls; i++)
            delete ctrls[i];

    }
    catch (const error& e) {
        fprintf(stderr, "Unexpected error: %s\n", e.what());
        return 1;
    }


    return 0;

}
