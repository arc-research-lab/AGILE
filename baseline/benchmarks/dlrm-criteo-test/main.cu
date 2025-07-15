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
        unsigned int * h_input_sparse, * d_input_sparse;
        unsigned int * h_vector_sum, * d_vector_sum;

        h_input_sparse = (unsigned int *)malloc(cfg.iteration * cfg.batch_size * cfg.sparse_input_dim * sizeof(unsigned int));
        loadBinFile(cfg.sparse_input_file, h_input_sparse, cfg.iteration * cfg.batch_size * cfg.sparse_input_dim * sizeof(unsigned int), 0);
        cuda_err_chk(cudaMalloc((void **)&d_input_sparse, cfg.iteration * cfg.batch_size * cfg.sparse_input_dim * sizeof(unsigned int)));
        cuda_err_chk(cudaMemcpy(d_input_sparse, h_input_sparse, cfg.iteration * cfg.batch_size * cfg.sparse_input_dim * sizeof(unsigned int), cudaMemcpyHostToDevice));

        h_vector_sum = (unsigned int *)malloc(cfg.iteration * cfg.batch_size * cfg.embedding_vec_dim * sizeof(unsigned int));
        cuda_err_chk(cudaMalloc((void **)&d_vector_sum, cfg.iteration * cfg.batch_size * cfg.embedding_vec_dim * sizeof(unsigned int)));

        std::chrono::high_resolution_clock::time_point start0, end0;
        start0 = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < cfg.iteration; ++i){
            unsigned int * d_sparse_in = d_input_sparse + i * cfg.batch_size * cfg.sparse_input_dim;
            unsigned int * d_vector_sum_off = d_vector_sum + i * cfg.batch_size * cfg.embedding_vec_dim;
            // load embedding
            sum_embedding<128>(enable_load, stream_compute, nvme_mem.d_array_ptr, d_sparse_in, d_vector_sum_off, 0, cfg.batch_size, cfg.sparse_input_dim, cfg.embedding_vec_dim);
        }
        cuda_err_chk(cudaStreamSynchronize(stream_compute));
        end0 = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end0 - start0);
        std::cout << "Time taken by function: "
            << duration.count() << " milliseconds" << std::endl;
        std::chrono::duration<double> time_span0 = std::chrono::duration_cast<std::chrono::duration<double>>(end0 - start0);

        unsigned int total_cmd =  nvme_mem.print_reset_stats();
        double bandwidth = total_cmd * cfg.slot_size;
        bandwidth /= 1024 * 1024 * 1024;
        bandwidth /= time_span0.count();
        std::cout << "Total time: " << time_span0.count() << " s\n";
        std::cout << "Total cmd: " << total_cmd << "\n";
        std::cout << "Bandwidth: " << bandwidth << " GB/s\n";

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
        


        for (size_t i = 0 ; i < cfg.n_ctrls; i++)
            delete ctrls[i];

    }
    catch (const error& e) {
        fprintf(stderr, "Unexpected error: %s\n", e.what());
        return 1;
    }


    return 0;

}
