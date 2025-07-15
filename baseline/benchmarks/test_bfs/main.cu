
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
__global__
void kernel_baseline_pc(array_d_t<uint32_t>* da, uint32_t *label, const uint32_t level, const uint32_t vertex_count,
                        const uint32_t *vertexList, uint32_t *changed) {
    const uint32_t tid = blockDim.x * BLOCK_NUM * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;

    if(tid < vertex_count && label[tid] == level) {
        const uint32_t start = vertexList[tid];
        const uint32_t end = vertexList[tid+1];

        // printf("start %d %d\n", start, end);
        for(uint32_t i = start; i < end; i++) {
            
            EdgeT next = da->seq_read(i);
            
            if(label[next] == MYINFINITY) {
                // printf("%d\n", next);
                label[next] = level + 1;
                *changed = true;
            }
        }
    }
}

__device__ unsigned int level_nodes = 0;


int main(int argc, char** argv) {

    int minGridSize = 0;
    int blockSize = 0;  // Will be calculated
    size_t sharedMemory = 4;  // Shared memory per block in bytes

    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, 
        &blockSize, 
        kernel_baseline_pc, 
        sharedMemory, 
        0
    );

    std::cout << "minGridSize: " << minGridSize << std::endl;
    std::cout << "blockSize: " << blockSize << std::endl;


    Settings settings;
    try
    {
        settings.parseArguments(argc, argv);
        // std::cout << "settings.cudaDevice: " << settings.cudaDevice << std::endl;
        // std::cout << "settings.input: " << settings.input << std::endl;
        // std::cout << "settings.nn: " << settings.node_num << std::endl;
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
        cuda_err_chk(cudaSetDevice(settings.cudaDevice));
        std::vector<Controller*> ctrls(settings.n_ctrls);
        for (size_t i = 0 ; i < settings.n_ctrls; i++)
            ctrls[i] = new Controller(settings.ssdtype == 0 ? sam_ctrls_paths[i] : intel_ctrls_paths[i], settings.nvmNamespace, settings.cudaDevice, settings.queueDepth, settings.numQueues);

        uint32_t b_size = settings.blkSize;//64;
        uint32_t g_size = (settings.numThreads + b_size - 1)/b_size;//80*16;
        uint32_t n_threads = b_size * g_size;

        uint32_t page_size = settings.pageSize;
        uint64_t n_pages = settings.numPages;
        uint64_t total_cache_size = (page_size * n_pages);
        std::cout << "page size: " << page_size << std::endl;
        std::cout << "page num: " << n_pages << std::endl;
        std::cout << "total cache size: " << total_cache_size << std::endl;
        page_cache_t h_pc(page_size, n_pages, settings.cudaDevice, ctrls[0][0], (uint32_t) 64, ctrls);
        
        //QueuePair* d_qp;
        page_cache_t* d_pc = (page_cache_t*) (h_pc.d_pc_ptr);
        #define TYPE uint32_t
        uint64_t n_elems = settings.edge_num; // 68993773; // how many elements 68993773
        uint64_t t_size = n_elems * sizeof(TYPE);

        range_t<uint32_t> h_range((uint32_t)0, (uint64_t)n_elems, (uint64_t)0, (uint64_t)(t_size/page_size) + 1, (uint64_t)0, (uint64_t)page_size, &h_pc, settings.cudaDevice);
        range_t<uint32_t>* d_range = (range_t<uint32_t>*) h_range.d_range_ptr;

        std::vector<range_t<uint32_t>*> vr(1);
        vr[0] = & h_range;
        array_t<uint32_t> nvme_mem(n_elems, 0, vr, settings.cudaDevice);

        std::cout << "finished creating range\n";
        std::cout << "atlaunch kernel\n";
        char st[15];
        cuda_err_chk(cudaDeviceGetPCIBusId(st, 15, settings.cudaDevice));
        std::cout << st << std::endl;

        uint32_t *label_d, *label_s;
        uint32_t *vertexList_d, *vertexList_s;
        uint32_t *changed_d, changed_s = false;
        cuda_err_chk(cudaMalloc(&changed_d, sizeof(uint32_t)));

        uint64_t numblocks, numthreads, vertex_count;
        vertex_count = settings.node_num;
        numthreads = settings.numThreads;
        
        numblocks = ((vertex_count + numthreads) / numthreads);
        // dim3 blockDim(BLOCK_NUM, (numblocks+BLOCK_NUM)/BLOCK_NUM);
        unsigned int blockDim = vertex_count / numthreads + 1;
        std::cout << "block num: " << std::dec << BLOCK_NUM << std::endl;
        std::cout << "----> threads per block: " << std::dec << numthreads << std::endl;

        cuda_err_chk(cudaMalloc(&label_d, (vertex_count) * sizeof(uint32_t)));
        label_s = (uint32_t*) malloc((vertex_count) * sizeof(uint32_t));
        cuda_err_chk(cudaMalloc(&vertexList_d, (vertex_count+1) * sizeof(uint32_t)));
        vertexList_s = (uint32_t*) malloc((vertex_count+1) * sizeof(uint32_t));
        
        std::ifstream row_file(settings.input);
        
        unsigned int data;
        for(int i = 0; i < settings.node_num + 1; ++i){
            row_file.read(reinterpret_cast<char*>(&data), sizeof(unsigned int));
            vertexList_s[i] = data;
        }
        memset(label_s, MYINFINITY, (vertex_count)* sizeof(uint32_t));

        label_s[0] = 0;

        cuda_err_chk(cudaMemcpy(label_d, label_s, sizeof(uint32_t) * vertex_count, cudaMemcpyHostToDevice));
        cuda_err_chk(cudaMemcpy(vertexList_d, vertexList_s, sizeof(uint32_t) * vertex_count+1, cudaMemcpyHostToDevice));

        // int grid_dim, block_dim, total_threads, node_pre_thread;
        
        // cuda_err_chk(cudaOccupancyMaxPotentialBlockSize(&grid_dim, &block_dim, kernel_baseline2_pc, 0, 0));
        // total_threads = grid_dim * block_dim;

        std::cout << "blockDim: " << blockDim << std::endl;
        std::cout << "numthreads: " << numthreads << std::endl;

        // node_pre_thread = std::ceil(((float) vertex_count) / ((float) total_threads));
        // void *args[] = {&nvme_mem.d_array_ptr, &label_d, &vertex_count, &vertexList_d, &changed_d, &vertex_count, &node_pre_thread};
        // kernel_baseline2_pc(array_d_t<uint32_t>* da, uint32_t *label, const uint32_t vertex_count,
        //                 const uint32_t *vertexList, uint32_t *changed, unsigned int total_nodes, unsigned int node_per_thread)

        // init_clk_logger(1024, 1024);
        
        int itr = 0;
        double total_itr_time = 0;
        std::chrono::high_resolution_clock::time_point start, end, s0, e0;
        cudaStream_t stream;
        start = std::chrono::high_resolution_clock::now();
        // cuda_err_chk(cudaLaunchCooperativeKernel((void*)kernel_baseline2_pc, grid_dim, block_dim, args, 0, stream));
        // cuda_err_chk(cudaDeviceSynchronize());
        do{
            
            changed_s = false;
            cuda_err_chk(cudaMemcpy(changed_d, &changed_s, sizeof(uint32_t), cudaMemcpyHostToDevice));
            s0 = std::chrono::high_resolution_clock::now();
            kernel_baseline_pc<<<(blockDim), numthreads>>>(nvme_mem.d_array_ptr, label_d, itr, vertex_count, vertexList_d, changed_d);
            cuda_err_chk(cudaDeviceSynchronize());
            e0 = std::chrono::high_resolution_clock::now();
            cuda_err_chk(cudaMemcpy(&changed_s, changed_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            double itr_time = std::chrono::duration_cast<std::chrono::duration<double>>(e0 - s0).count();
            total_itr_time += itr_time;
            // printf("level %d\n", itr);
            std::cout << "summary, level, " << itr << ", time: " << itr_time << " ns\n";
            ++itr;
        }while(changed_s);
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span0 = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        std::cout << "BFS time: " << total_itr_time << " seconds." << std::endl;
        cuda_err_chk(cudaMemcpy(label_s, label_d, sizeof(uint32_t) * vertex_count, cudaMemcpyDeviceToHost));
        cuda_err_chk(cudaDeviceSynchronize());

        // dump_event_log("nvme_log.csv");

        std::ofstream bfs_res("gpu-res-bfs.bin", std::ios::out | std::ios::binary);
        for(int i = 0; i < vertex_count; ++i){
            bfs_res.write(reinterpret_cast<const char*>(label_s + i), sizeof(unsigned int));
        }

        bfs_res.close();

        unsigned int total_cmd =  nvme_mem.print_reset_stats();
        double bandwidth = total_cmd * settings.pageSize;
        bandwidth /= total_itr_time;
        std::cout << "Bandwidth: " << bandwidth << " GB/s\n";

        for (size_t i = 0 ; i < settings.n_ctrls; i++)
            delete ctrls[i];
    }
    catch (const error& e) {
        fprintf(stderr, "Unexpected error: %s\n", e.what());
        return 1;
    }


    return 0;

}
