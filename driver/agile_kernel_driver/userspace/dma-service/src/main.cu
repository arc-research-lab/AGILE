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
#include "t_agile_service.h"
#include "agile_helper.h"

#include "config.h"

#include "io_utils.h"

__global__ 
void dma_issue_kernel(AgileDmaEngine * engine, size_t size, uint32_t *src_idx, uint32_t *dst_idx, char flags, uint32_t total_cmd, uint32_t repeat){

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    AgileLockChain chain;
    uint32_t *lock;
    for(uint32_t rep = 0; rep < repeat; ++rep){
        for(uint32_t i = tid; i < total_cmd; i += blockDim.x * gridDim.x){
            // Submit DMA request
            engine->submit(i * size, i * size, size, flags, &lock, &chain);
            wait_cmd(lock);
        }
    }
}

int main(int argc, char ** argv){

    Configs conf(argc, argv);
    conf.parse();

    // Initialize CUDA
    ASSERTDRV(cuInit(0));
    CUdevice dev;
    ASSERTDRV(cuDeviceGet(&dev, 0));

    CUcontext dev_ctx;
    ASSERTDRV(cuDevicePrimaryCtxRetain(&dev_ctx, dev));
    ASSERTDRV(cuCtxSetCurrent(dev_ctx));

    INFO("main starting...");

    AgileDriver driver; // can only have one instance.

    INFO("Total", driver.getAvaiableCpuDma(), "DMA engines available.");
    driver.setDmaQueuePairCudaDma(0, 128);
    driver.setDmaQueuePairCpuDma(1, 128);
    driver.setMonitorThreadsNum(1);
    driver.setWorkerThreadsNum(1);
    driver.setHbmCacheSize(1024l * 1024l * 1024l); // 1GB HBM cache
    driver.setDramCacheSize(1024l * 1024l * 1024l); // TODO: this is reserved in grub.

    driver.allocateHost();

    
    
    driver.startMonitors();
    driver.startWorkers();

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
        1024, // threads per bc       
        0
    ));

    cudaStream_t kernel_s;
    cudaStream_t service_s;
    cuda_err_chk(cudaStreamCreateWithFlags(&kernel_s, cudaStreamNonBlocking));
    cuda_err_chk(cudaStreamCreateWithFlags(&service_s, cudaStreamNonBlocking));

    AgileGpuMem reserved_hbm_mem = driver.getReservedMem();
    pollingService<<<1, 1024, 0, service_s>>>(driver.getAgileDmaQueuePairDevicePtr(), driver.getDmaQueuePairNum(), (uint32_t *)reserved_hbm_mem.d_ptr);
    auto start = std::chrono::high_resolution_clock::now();
    dma_issue_kernel<<<1, 32, 0, kernel_s>>>(driver.getDmaEngineDevice(), conf.transfer_size, nullptr, nullptr, DMA_CPU2GPU, conf.command_num, conf.repeat);
    cuda_err_chk(cudaStreamSynchronize(kernel_s));
    auto end = std::chrono::high_resolution_clock::now();

    stop_service((uint32_t *)reserved_hbm_mem.h_ptr);
    cuda_err_chk(cudaStreamSynchronize(service_s));

    double total_time = std::chrono::duration<double, std::milli>(end - start).count();
    double total_gb = ((double)conf.transfer_size * conf.command_num * conf.repeat) / (1024.0 * 1024.0 * 1024.0);
    INFO("Total data:", total_gb, "GB");
    INFO("Total time:", total_time, "ms");
    INFO("Throughput:", total_gb / (total_time / 1000.0), "GB/s");

    cuda_err_chk(cudaStreamDestroy(kernel_s));
    cuda_err_chk(cudaStreamDestroy(service_s));


    driver.stopMonitors();
    driver.stopWorkers();
    driver.freeHost();

    return 0;
}