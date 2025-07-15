
#include <nvm_ctrl.h>
#include <nvm_types.h>
#include <nvm_queue.h>
#include <nvm_util.h>
#include <nvm_admin.h>
#include <nvm_error.h>
#include <nvm_cmd.h>
#include <string>
#include <stdexcept>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <ctrl.h>
#include <buffer.h>
#include "settings.h"
#include <event.h>
#include <queue.h>
#include <nvm_parallel_queue.h>

#include <nvm_io.h>
#include <page_cache.h>
#include <util.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <cmath>

#include <malloc.h>
#include <memory.h>
#include <float.h>

#define __ATOMIC_THREAD 10

#include <cuda_runtime_api.h> 
#include <cooperative_groups.h>
#include <cuda.h>

#ifdef __DIS_CLUSTER__
#include <sisci_api.h>
#endif

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

typedef uint32_t EdgeT;

// __global__
// void count_column(array_d_t<uint32_t>* da, unsigned int * offsets, unsigned int * column_count, unsigned int nodes){
//     unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
//     if(tid < nodes){
//     // if(tid == 0){
//         unsigned int start = offsets[tid];
//         unsigned int end = offsets[tid + 1];
//         printf("start: %d end: %d\n", start, end);
//         for(int j = start; j < end; ++j){
//             unsigned int col = da->seq_read(j);
//             printf("col: %d\n", col);
//             // atomicAdd_system(column_count + col, 1);
//         }
//     }
// }

// __global__
// void norm_column(array_d_t<uint32_t>* da, unsigned int * offsets, unsigned int * column_count, unsigned int nodes, unsigned int value_offset){
//     unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
//     if(tid < nodes){
//         unsigned int start = offsets[tid];
//         unsigned int end = offsets[tid + 1];
//         for(int j = start; j < end; ++j){
//             unsigned int col = da->seq_read(j);
//             unsigned int count = column_count[col];
//             float val = 1.0 / count;
//             (*da)(value_offset + j, __float_as_uint(val));
//         }
//     }
// }

__global__
void pagerank_itr(array_d_t<uint32_t>* da, unsigned int * offsets, unsigned int nodes, unsigned int value_offset, float * vec, float * output_vec, float damping, float * norm2){
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < nodes){
    // if(tid < 10){
        output_vec[tid] = 0;
        unsigned int start = offsets[tid];
        unsigned int end = offsets[tid + 1];
        // printf("%d %d\n", start, end);
        for(unsigned int j = start; j < end; ++j){
            unsigned int col = da->seq_read(j);
            unsigned int val = da->seq_read(value_offset + j);
            // if(col > nodes){
            //     printf("error %d\n", col);
            // }else
            output_vec[tid] += __uint_as_float(val) * vec[col];
        }
        output_vec[tid] *= damping;
        output_vec[tid] += (1.0 - damping) / nodes;
        float diff = output_vec[tid] - vec[tid];
        atomicAdd_system(norm2,  diff * diff);
    }
}

template<typename T>
void save_data_txt(std::string path, T * arr, unsigned int size){
    std::ofstream distance_bin(path, std::ios::out);
    for(int i = 0; i < size; ++i){
        // distance_bin.write(reinterpret_cast<const char*>(&arr[i]), sizeof(T));
        distance_bin << arr[i] << "\n";
    }
    distance_bin.close();
}

int main(int argc, char** argv) {

    int minGridSize = 0;
    int blockSize = 0;  // Will be calculated
    size_t sharedMemory = 0;  // Shared memory per block in bytes

    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, 
        &blockSize, 
        pagerank_itr, 
        sharedMemory, 
        0
    );

    std::cout << "minGridSize: " << minGridSize << std::endl;
    std::cout << "blockSize: " << blockSize << std::endl;

    Settings settings;
    try
    {
        settings.parseArguments(argc, argv);
    }
    catch (const string& e)
    {
        fprintf(stderr, "%s\n", e.c_str());
        fprintf(stderr, "%s\n", Settings::usageString(argv[0]).c_str());
        return 1;
    }


    cudaDeviceProp properties;
    if (cudaGetDeviceProperties(&properties, settings.cudaDevice) != cudaSuccess)
    {
        fprintf(stderr, "Failed to get CUDA device properties\n");
        return 1;
    }

    try {
        std::cout << "here\n";
        cuda_err_chk(cudaSetDevice(settings.cudaDevice));
        std::vector<Controller*> ctrls(settings.n_ctrls);
        for (size_t i = 0 ; i < settings.n_ctrls; i++)
            ctrls[i] = new Controller(settings.ssdtype == 0 ? sam_ctrls_paths[i] : intel_ctrls_paths[i], settings.nvmNamespace, settings.cudaDevice, settings.queueDepth, settings.numQueues);

        uint32_t b_size = settings.blkSize;//64;
        uint32_t g_size = (settings.numThreads + b_size - 1)/b_size;//80*16;
        uint32_t n_threads = b_size * g_size;

        uint32_t page_size = settings.pageSize;
        uint32_t n_pages = settings.numPages;
        uint32_t total_cache_size = (page_size * n_pages);
        std::cout << "page size: " << page_size << std::endl;
        std::cout << "page num: " << n_pages << std::endl;
        std::cout << "total cache size: " << total_cache_size << std::endl;
        page_cache_t h_pc(page_size, n_pages, settings.cudaDevice, ctrls[0][0], (uint32_t) 64, ctrls);
        
        //QueuePair* d_qp;
        page_cache_t* d_pc = (page_cache_t*) (h_pc.d_pc_ptr);
        #define TYPE uint32_t

        uint64_t total_ssd_pages = settings.total_ssd_pages * 4;
        uint64_t n_elems = total_ssd_pages * settings.pageSize / 4; // how many elements
        uint64_t t_size = n_elems * sizeof(TYPE);
        uint64_t ssd_page_offset = 0;
        unsigned int nodes = settings.node_num; // edges = 508836 // hubei
        std::string row_path = settings.input;
        unsigned int edge_offset = 0;
        unsigned int weight_offset = settings.total_ssd_pages * settings.pageSize / 4;

        range_t<uint32_t> h_range((uint32_t)0, (uint64_t)n_elems, (uint64_t)ssd_page_offset, (uint64_t) total_ssd_pages/*(t_size/page_size)*/, (uint64_t)0, (uint64_t)page_size, &h_pc, settings.cudaDevice);
        range_t<uint32_t>* d_range = (range_t<uint32_t>*) h_range.d_range_ptr;

        std::vector<range_t<uint32_t>*> vr(1);
        vr[0] = & h_range;
        array_t<uint32_t> nvme_mem(n_elems, 0, vr, settings.cudaDevice);

        std::cout << "finished creating range\n";
        std::cout << "atlaunch kernel\n";
        char st[15];
        cuda_err_chk(cudaDeviceGetPCIBusId(st, 15, settings.cudaDevice));
        std::cout << st << std::endl;

        

        
        // cuda_err_chk(cudaMalloc(&changed_d, sizeof(unsigned int)));
        // unsigned int itr = 0;
        unsigned int block_dim = settings.numThreads;
        unsigned int grid_dim = nodes / block_dim + 1;

        unsigned int * offsets_h = (unsigned int *) malloc(sizeof(unsigned int) * (nodes + 1));
        unsigned int * offsets_d;
        cuda_err_chk(cudaMalloc(&offsets_d, sizeof(unsigned int) * (nodes + 1)));

        // std::ifstream rowtxt(row_path);
        // std::string line;
        // for(int i = 0; i < nodes + 1; ++i){
        //     rowtxt >> line;
        //     offsets_h[i] = atoi(line.c_str());
        // }

        std::ifstream row_file(settings.input);
        unsigned int data;
        for(unsigned int i = 0; i < settings.node_num + 1; ++i){
            row_file.read(reinterpret_cast<char*>(&data), sizeof(unsigned int));
            offsets_h[i] = data;
        }

        cuda_err_chk(cudaMemcpy(offsets_d, offsets_h, sizeof(unsigned int) * (nodes + 1), cudaMemcpyHostToDevice));

        unsigned int * column_count_d;
        float * vec_h, * vec_d, * output_vec_d, * norm2_d, norm2;
        vec_h = (float *) malloc(sizeof(float) * nodes);
        cuda_err_chk(cudaMalloc(&column_count_d, sizeof(unsigned int) * (nodes)));
        cuda_err_chk(cudaMalloc(&vec_d, sizeof(float) * (nodes)));
        cuda_err_chk(cudaMalloc(&output_vec_d, sizeof(float) * (nodes)));
        cuda_err_chk(cudaMalloc(&norm2_d, sizeof(float) * (nodes)));

        float initial_val = 1.0f / nodes;
        for(unsigned int i = 0; i < nodes; ++i){
            vec_h[i] = initial_val;
        }
        cuda_err_chk(cudaMemcpy(vec_d, vec_h, sizeof(unsigned int) * (nodes), cudaMemcpyHostToDevice));
        
        std::cout << "start kernel\n";
        std::chrono::high_resolution_clock::time_point start, end, s0, e0;

        

        // count_column<<<grid_dim, block_dim>>>(nvme_mem.d_array_ptr, offsets_d, column_count_d, nodes);
        // cuda_err_chk(cudaDeviceSynchronize());
        // norm_column<<<grid_dim, block_dim>>>(nvme_mem.d_array_ptr, offsets_d, column_count_d, nodes, weight_offset);
        // cuda_err_chk(cudaDeviceSynchronize());
        // // d_pc->flush_cache();
        unsigned int itr = 0;
        double total_itr_time = 0;
        start = std::chrono::high_resolution_clock::now();
        do{
            
            norm2 = 0;
            cuda_err_chk(cudaMemcpy(norm2_d, &norm2, sizeof(float), cudaMemcpyHostToDevice));

            s0 = std::chrono::high_resolution_clock::now();
            pagerank_itr<<<grid_dim, block_dim>>>(nvme_mem.d_array_ptr, offsets_d, nodes, weight_offset, vec_d, output_vec_d, 0.85, norm2_d);
            cuda_err_chk(cudaDeviceSynchronize());
            e0 = std::chrono::high_resolution_clock::now();


            cuda_err_chk(cudaMemcpy(vec_d, output_vec_d, sizeof(float) * (nodes), cudaMemcpyDeviceToDevice));
            cuda_err_chk(cudaMemcpy(&norm2, norm2_d, sizeof(float), cudaMemcpyDeviceToHost));
            norm2 = sqrt(norm2);
            
            double itr_time = std::chrono::duration_cast<std::chrono::duration<double>>(e0 - s0).count();
            total_itr_time += itr_time;
            std::cout << "itr: " << std::dec << itr << " norm: " << norm2  << " time: " << itr_time << " ns"<< std::endl;
            itr++;
            
        }while(itr < settings.max_itr);
        // }while(!(itr > settings.max_itr || norm2 > settings.error_thresh));
        
        
        end = std::chrono::high_resolution_clock::now();
        
        std::cout << "total time " << total_itr_time << " ns" << std::endl;

        unsigned int total_cmd =  nvme_mem.print_reset_stats();
        double bandwidth = total_cmd * settings.pageSize;
        bandwidth /= total_itr_time;
        std::cout << "Bandwidth: " << bandwidth << " GB/s\n";
        
        cuda_err_chk(cudaMemcpy(vec_h, vec_d, (nodes) * sizeof(float), cudaMemcpyDeviceToHost));
        save_data_txt(settings.output, vec_h, nodes);


        for (size_t i = 0 ; i < settings.n_ctrls; i++)
            delete ctrls[i];
    }
    catch (const error& e) {
        fprintf(stderr, "Unexpected error: %s\n", e.what());
        return 1;
    }


}
