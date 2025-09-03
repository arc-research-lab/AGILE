#include <iostream>
#include <thread>
#include <chrono>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include <sys/syscall.h>
#include <unistd.h>

#include "agile_driver.h"
#include "agile_dmem.h"
#include "t_agile_service.h"
#include "agile_helper.h"

#include "io_utils.h"

__global__ 
void dma_issue_kernel(AgileDmaEngine * engine, size_t size, uint32_t *src_idx, uint32_t *dst_idx, char flags, uint32_t total_cmd, uint32_t repeat){

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    AgileLockChain chain;
    for(uint32_t rep = 0; rep < repeat; ++rep){
        for(uint32_t i = tid; i < total_cmd; i += blockDim.x * gridDim.x){
            // Submit DMA request
            // uint64_t src_addr = 0 + ((uint64_t)src_idx[i] * size);
            // uint64_t dst_addr = 0 + ((uint64_t)dst_idx[i] * size);
            
            cmd_info_t info;
            engine->submit(i * size, i * size, size, flags, info, &chain);
            engine->wait(info);
            // printf("cmd idx: %d\n", i);
        }
    }
    // __syncthreads();
    // if(tid == 0){
    //     printf("finish\n");
    // }
}

int main(int argc, char ** argv){

    if(argc < 4){
        printf("argv[1]: transfer_size, argv[2]: total_cmd, argv[3]: repeat\n");
        return 0;
    }

    cudaSetDevice(0);
    cudaFree(0);      
    uint32_t transfer_size = atoi(argv[1]);
    uint32_t total_cmd = atoi(argv[2]);
    uint32_t max_bid = 1024*1024*1024/transfer_size;
    uint32_t repeat = atoi(argv[3]);

    pined_gpu_mem pined_mem;
    allocateGPUPinedMem(0, 1024 * 1024 * 1024 * 2, pined_mem);
    INFO("main starting...");

    void * host_ptr = malloc(1024*1024*1024);
    if(!host_ptr){
        printf("malloc error\n");
        exit(0);
    }
    AgileDriver driver;
    driver.open();
    driver.setPinedGpuMem(&pined_mem);
    driver.setManagedPtr(host_ptr);
    int dma_cmd_queue_num = 1;
    
    driver.setDmaCmdQueueNum(dma_cmd_queue_num);
    driver.setDmaCmdQueueDepth(256);
    driver.setMonitorNum(dma_cmd_queue_num);
    driver.setWorkerNum(1);
    driver.allocateHost();

    driver.startWorker();
    driver.startMonitor();
    
    shared_hbm_t * d_shared_hbm = driver.getSharedHbmDevicePtr();
    shared_hbm_t * h_shared_hbm = driver.getSharedHbmHostPtr();
    
    int *src_ptr = (int *) driver.getHostMemPtr();
    int *dst_ptr = (int *) driver.getDeviceMemPtr();
    int *h_dst_ptr = (int *) driver.getDeviceMemMappedPtr();

    uint32_t *h_src_idx, *d_src_idx;
    uint32_t *h_dst_idx, *d_dst_idx;
    h_src_idx = (uint32_t *) malloc(sizeof(uint32_t) * total_cmd);
    h_dst_idx = (uint32_t *) malloc(sizeof(uint32_t) * total_cmd);
    cuda_err_chk(cudaMalloc(&d_src_idx, sizeof(uint32_t) * total_cmd));
    cuda_err_chk(cudaMalloc(&d_dst_idx, sizeof(uint32_t) * total_cmd));

    for(uint32_t i = 0; i < total_cmd; ++i){
        h_src_idx[i] = i;
        h_dst_idx[i] = i;
    }

    for(uint32_t i = 0; i < transfer_size / sizeof(int) * total_cmd; ++i){
        src_ptr[i] = 0;
        h_dst_ptr[i] = i;
    }

    cuda_err_chk(cudaMemcpy(d_src_idx, h_src_idx, sizeof(uint32_t) * total_cmd, cudaMemcpyHostToDevice));
    cuda_err_chk(cudaMemcpy(d_dst_idx, h_dst_idx, sizeof(uint32_t) * total_cmd, cudaMemcpyHostToDevice));
    cuda_err_chk(cudaDeviceSynchronize());

    int numBlocksPerSM = 0;

    cuda_err_chk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM,
        pollingService,
        1024, // threads per bc       
        0
    ));

    cuda_err_chk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM,
        dma_issue_kernel,
        1024, // threads per block
        0    // dynamic shared memory
    ));

    cudaStream_t kernel_s;
    cudaStream_t service_s;
    cuda_err_chk(cudaStreamCreateWithFlags(&kernel_s, cudaStreamNonBlocking));
    cuda_err_chk(cudaStreamCreateWithFlags(&service_s, cudaStreamNonBlocking));

    pollingService<<<1, 1024, 0, service_s>>>(driver.getAgileDmaQueuePairDevicePtr(), dma_cmd_queue_num, &(d_shared_hbm->stop_sig));
    using namespace std::chrono_literals; // Enables literals like 500ms
    std::this_thread::sleep_for(500ms);

    // measure kernel execution time
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    dma_issue_kernel<<<1, 1024, 0, kernel_s>>>(driver.getDmaEngine()->getDevicePtr(), transfer_size, d_src_idx, d_dst_idx, DMA_GPU2CPU, total_cmd, repeat);
    debug("Waiting...");
    cuda_err_chk(cudaStreamSynchronize(kernel_s));
    std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();
    debug("Kernel Finish");
    // for(uint32_t i = 0; i < transfer_size / sizeof(int) * total_cmd; ++i){
    //     printf("%d ", src_ptr[i]);
    // }
    // printf("\n");

    auto duration =  std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);

    std::cout << "Total time: " << duration.count() << " nanoseconds" << std::endl;
    double total_b = ((double)total_cmd * (double)transfer_size) * repeat;
    std::cout << "Total data transferred: " << total_b << " Bytes" << std::endl;
    double bandwidth = total_b / (duration.count());
    std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;
    driver.stopMonitor();
    driver.stopWorker();
    uint32_t * stop_sig = (uint32_t *) h_shared_hbm;
    *stop_sig = 1;
    free(h_src_idx);
    free(h_dst_idx);
    free(host_ptr);
    cuda_err_chk(cudaStreamDestroy(kernel_s));
    cuda_err_chk(cudaStreamDestroy(service_s));
    driver.freeHost();
    driver.close();
    return 0;
}