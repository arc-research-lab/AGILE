#include <iostream>
#include <thread>
#include <chrono>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include <sys/syscall.h>
#include <unistd.h>

#include <boost/program_options.hpp>
#include <fstream>

#include "agile_driver.h"
#include "agile_dmem.h"
#include "t_agile_service.h"
#include "agile_helper.h"

#include "config.h"

#include "io_utils.h"

__global__ 
void dma_issue_kernel(AgileDmaEngine * engine, size_t size, uint32_t *src_idx, uint32_t *dst_idx, char flags, uint32_t total_cmd, uint32_t repeat){

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    AgileLockChain chain;
    for(uint32_t rep = 0; rep < repeat; ++rep){
        for(uint32_t i = tid; i < total_cmd; i += blockDim.x * gridDim.x){
            // Submit DMA request
            cmd_info_t info;
            engine->submit(i * size, i * size, size, flags, info, &chain);
            engine->wait(info);
        }
    }
}

int main(int argc, char ** argv){

    Configs conf(argc, argv);
    conf.parse();

    cudaSetDevice(0);
    cudaFree(0);      

    pined_gpu_mem pined_mem;
    allocateGPUPinedMem(0, 1024 * 1024 * 1024 * 2, pined_mem);
    INFO("main starting...");

    void * host_ptr = malloc(1024 * 1024 * 1024);
    if(!host_ptr){
        printf("malloc error\n");
        exit(0);
    }

    AgileDriver driver;
    driver.open();
    driver.setPinedGpuMem(&pined_mem);
    driver.setManagedPtr(host_ptr);
    
    driver.setDmaCmdQueueNum(conf.queue_num);
    driver.setDmaCmdQueueDepth(conf.queue_depth);
    driver.setMonitorNum(conf.monitor_threads);
    driver.setWorkerNum(conf.worker_threads);
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
    h_src_idx = (uint32_t *) malloc(sizeof(uint32_t) * conf.command_num);
    h_dst_idx = (uint32_t *) malloc(sizeof(uint32_t) * conf.command_num);
    cuda_err_chk(cudaMalloc(&d_src_idx, sizeof(uint32_t) * conf.command_num));
    cuda_err_chk(cudaMalloc(&d_dst_idx, sizeof(uint32_t) * conf.command_num));

    for(uint32_t i = 0; i < conf.command_num; ++i){
        h_src_idx[i] = i;
        h_dst_idx[i] = i;
    }

    for(uint32_t i = 0; i < conf.transfer_size / sizeof(int) * conf.command_num; ++i){
        src_ptr[i] = 0;
        h_dst_ptr[i] = i;
    }

    cuda_err_chk(cudaMemcpy(d_src_idx, h_src_idx, sizeof(uint32_t) * conf.command_num, cudaMemcpyHostToDevice));
    cuda_err_chk(cudaMemcpy(d_dst_idx, h_dst_idx, sizeof(uint32_t) * conf.command_num, cudaMemcpyHostToDevice));
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

    pollingService<<<1, 1024, 0, service_s>>>(driver.getAgileDmaQueuePairDevicePtr(), conf.queue_num, &(d_shared_hbm->stop_sig));
    using namespace std::chrono_literals; // Enables literals like 500ms
    std::this_thread::sleep_for(500ms);

    // measure kernel execution time
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    dma_issue_kernel<<<1, 1024, 0, kernel_s>>>(driver.getDmaEngine()->getDevicePtr(), conf.transfer_size, d_src_idx, d_dst_idx, DMA_GPU2CPU, conf.command_num, conf.repeat);
    debug("Waiting...");
    cuda_err_chk(cudaStreamSynchronize(kernel_s));
    std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();
    debug("Kernel Finish");

    auto duration =  std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);

    std::cout << "Total time: " << duration.count() << " nanoseconds" << std::endl;
    double total_b = ((double)conf.command_num * (double)conf.transfer_size) * conf.repeat;
    std::cout << "Total data transferred: " << total_b << " Bytes" << std::endl;
    double bandwidth = total_b / (duration.count());
    std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;
    driver.stopMonitor();
    driver.stopWorker();

    auto monitors = driver.getHostMonitors();
    // for(uint32_t i = 0; i < m_threads; ++i){
    //     char filename[256];
    //     sprintf(filename, "log_monitor_%d.txt", i);
    //     monitors[i]->printLogs(filename);
    // }
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