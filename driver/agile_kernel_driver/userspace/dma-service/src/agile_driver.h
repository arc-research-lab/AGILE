#pragma once

#include <iostream>
#include <vector>
#include <atomic>
#include <map>
#include <sys/mman.h>
#include <sys/epoll.h> 
#include <sys/eventfd.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>

#include "agile_gpu_krnl.h"
#include "agile_kernel_driver.h"
#include "agile_dma_service.h"


#include "io_utils.h"

typedef unsigned long long u64;
#define PAGE_ROUND_UP(x, n)     (((x) + ((n) - 1)) & ~((n) - 1))

class AgileGpuMem {
public:
    AgileGpuMem(){}
    AgileGpuMem(pin_buffer_params params) : d_ptr(nullptr), h_ptr(nullptr), size(0), buffer_params(params) {}
    void * d_ptr; 
    void * h_ptr;
    uint64_t phy_addr;
    size_t size; // size after alignment
    pin_buffer_params buffer_params; 
};

class AgileDriver {

    bool open(){

        fd_gpu = ::open("/dev/AGILE-gpu", O_RDWR);
        if(fd_gpu < 0){
            perror("Failed to open /dev/AGILE-gpu");
        }

        fd_drv = ::open("/dev/AGILE-kernel", O_RDWR);
        if (fd_drv < 0) {
            perror("Failed to open /dev/AGILE-kernel");
            return false;
        }
        if (ioctl(fd_drv, IOCTL_GET_TOTAL_DMA_CHANNELS, &total_dma_channels) < 0) {
            perror("ioctl: get total DMA channels");
            close();
            return false;
        }

        fd_mem = ::open("/dev/AGILE-reserved_mem", O_RDWR);
        
        if (fd_mem < 0) {
            perror("Failed to open /dev/AGILE-reserved_mem");
            close();
            return false;
        }
        reserved_mem_ptr = ::mmap(NULL, reserve_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_mem, 0);
        if (reserved_mem_ptr == MAP_FAILED) {
            perror("mmap");
            close();
            return false;
        }
        
        cuda_err_chk(cudaHostRegister(this->reserved_mem_ptr, reserve_size, cudaHostRegisterIoMemory));
        
        return true;
    }

    bool close(){


        if (fd_drv >= 0) {
            ::close(fd_drv);
            fd_drv = -1;
        }

        if (reserved_mem_ptr) {
            cuda_err_chk(cudaHostUnregister(reserved_mem_ptr));
            ::munmap(reserved_mem_ptr, reserve_size);
            reserved_mem_ptr = NULL;
        }

        if (fd_mem >= 0) {
            ::close(fd_mem);
            fd_mem = -1;
        }

        if (fd_gpu >= 0){
            ::close(fd_gpu);
            fd_gpu = -1;
        }
        return true;
    }

public:

    AgileDriver(){
        engine = nullptr;
        this->open();
    }

    ~AgileDriver(){
        // free allocated gpu buffer
        for(uint32_t i = 0; i < allocated_buffers.size(); ++i){

        }
        this->close();
    }

    bool allocateHostDmaBuffer(struct dma_buffer * buf, size_t size){
        if (!buf) {
            std::cerr << "Invalid DMA buffer." << std::endl;
            return false;
        }
        if(buf->vaddr_user){
            std::cerr << "DMA buffer is already allocated." << std::endl;
            return false;
        }
        buf->size = size;
        if (ioctl(fd_drv, IOCTL_ALLOCATE_CACHE_BUFFER, buf) < 0) {
            perror("ioctl: allocate cache buffer");
            close();
            return false;
        }
        buf->vaddr_user = mmap(NULL, buf->size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_drv, 0);
        if (buf->vaddr_user == MAP_FAILED) {
            perror("mmap");
            ioctl(fd_drv, IOCTL_FREE_CACHE_BUFFER, buf);
            close();
            return false;
        }
        
        memset(buf->vaddr_user, 0, buf->size);

        cuda_err_chk(cudaSetDeviceFlags(cudaDeviceMapHost));
        auto err = cudaHostRegister(buf->vaddr_user, buf->size, cudaHostRegisterMapped);
        // void* dptr = nullptr;
        cuda_err_chk(cudaHostGetDevicePointer(&buf->vaddr_cuda, buf->vaddr_user, 0)); // TODO: get device pointer when needed.
        printf("%s:%d host ptr %p, kernel ptr %p\n", __FILE__, __LINE__, buf->vaddr_user, buf->vaddr_krnl);
        if(err != 0){
            freeHostDmaBuffer(buf);
        }
        return true;
    }

    bool freeHostDmaBuffer(struct dma_buffer * buf){
        if (buf->vaddr_user) {
            cuda_err_chk(cudaHostUnregister(buf->vaddr_user));
            munmap(buf->vaddr_user, buf->size);
            buf->vaddr_user = NULL;
        }
        if (ioctl(fd_drv, IOCTL_FREE_CACHE_BUFFER, buf) < 0) {
            perror("ioctl: free cache buffer");
            return false;
        }
        return true;
    }

    AgileGpuMem* allocatePinGpuBuf(uint64_t allocated_size) {

        CUdeviceptr ptr, aligned_ptr;
        ASSERTDRV(cuMemAlloc(&ptr, allocated_size));
        ASSERTDRV(cuPointerSetAttribute(&ptr, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, ptr));
        aligned_ptr = PAGE_ROUND_UP(ptr, GPU_PAGE_SIZE);
        
        pin_buffer_params params;
        params.vaddr = aligned_ptr;
        params.size = allocated_size - (aligned_ptr - ptr);
        params.p2p_token = 0;
        params.va_space = 0;

        AgileGpuMem *mem = new AgileGpuMem(params);
        mem->d_ptr = (void*)aligned_ptr;
        mem->size = params.size;
        if(ioctl(fd_gpu, IOCTL_PIN_GPU_BUFFER, &mem->buffer_params) < 0){
            perror("ioctl");
            delete mem;
            return nullptr;
        }
        mem->phy_addr = mem->buffer_params.phy_addr;
        mem->h_ptr = mmap(nullptr, mem->size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_gpu, mem->phy_addr);
        allocated_buffers.push_back(mem);
        return mem;
    }

    void freeGpuBuf(AgileGpuMem* mem) {
        auto it = std::find(allocated_buffers.begin(), allocated_buffers.end(), mem);
        if (it != allocated_buffers.end()) {
            munmap(mem->h_ptr, mem->size);
            // unpin GPU buffer
            if(ioctl(fd_gpu, IOCTL_UNPIN_GPU_BUFFER, &mem->buffer_params) < 0){
                perror("ioctl");
            }
            ASSERTDRV(cuMemFree((CUdeviceptr)mem->buffer_params.vaddr));
            allocated_buffers.erase(it);
            delete mem;
        }
    }

    void startWorkers(){
        for(uint32_t i = 0; i < worker_num; ++i){
            worker_threads.emplace_back([this, i](){
                // pin_current_thread_to(i*2 + 8);
                int cpu_id = sched_getcpu();
                int node = numa_node_of_cpu(cpu_id);
                INFO("start working on CPU", cpu_id, "NUMA node", node);
                this->host_workers[i]->start();
            });
        }
    }

    void stopWorkers(){
        for(uint32_t i = 0; i < worker_num; ++i){
            // write some data to stop_signals
            this->host_workers[i]->stop();
        }
        for(uint32_t i = 0; i < worker_num; ++i){
            worker_threads[i].join();
        }
    }

    void startMonitors(){
       
        for(uint32_t i = 0; i < monitor_num; ++i){
            host_monitors[i]->start();
        }
    }

    void stopMonitors(){
        for(uint32_t i = 0; i < monitor_num; ++i){
            host_monitors[i]->stop();
        }
    }

    AgileHostMonitor ** getHostMonitors(){
        return this->host_monitors;
    }

    AgileHostWorker ** getHostWorkers(){
        return this->host_workers;
    }

    int getAvaiableCpuDma(){
        return total_dma_channels;
    }


    void setDmaQueuePairCudaDma(uint32_t q_num,uint32_t q_depth){
        this->cuda_dma_queue_num = q_num;
        this->cuda_dma_queue_depth = q_depth;
    }

    void setDmaQueuePairCpuDma(uint32_t q_num,uint32_t q_depth){
        this->cpu_dma_queue_num = q_num;
        this->cpu_dma_queue_depth = q_depth;
    }

    /**
     * Poll new commands from both CUDA DMA queue and CPU DMA queue
     */
    void setMonitorThreadsNum(uint32_t threads_num){
        this->monitor_num = threads_num;
    }

    /**
     * Only for CUDA DMA queue
     */
    void setWorkerThreadsNum(uint32_t threads_num){
        this->worker_num = threads_num;
    }

    void allocateHost(){
        this->dma_cmd_queue_num = this->cuda_dma_queue_num + this->cpu_dma_queue_num;
        if(this->dma_cmd_queue_num == 0){
            ERROR("No DMA queue is configured");
            return;
        }
        
        // allocate GPU memory for CQs
        uint64_t gpu_mem_size = sizeof(struct agile_dma_cpl_t) * (this->cuda_dma_queue_num * this->cuda_dma_queue_depth + this->cpu_dma_queue_num * this->cpu_dma_queue_depth);
        uint64_t gpu_mem_size_aligned = PAGE_ROUND_UP(gpu_mem_size, 65536l);
        DEBUG("require contigous HBM", gpu_mem_size, "Bytes, aligned to", gpu_mem_size_aligned, "Bytes");
        this->cpl_buf = this->allocatePinGpuBuf(gpu_mem_size_aligned);
        this->cpl_reserved_size = gpu_mem_size_aligned - gpu_mem_size;
        this->cpl_reserved_offset = gpu_mem_size;

        // allocate contiguous DRAM for SQs
        uint64_t cpu_mem_size = sizeof(struct agile_dma_cmd_t) * (this->cuda_dma_queue_num * this->cuda_dma_queue_depth + this->cpu_dma_queue_num * this->cpu_dma_queue_depth);
        uint64_t cpu_mem_size_aligned = PAGE_ROUND_UP(cpu_mem_size, 65536l); // The DRAM buffer should be aligned to 64KB as required by GPU.
        DEBUG("require contigous DRAM", cpu_mem_size, "Bytes, aligned to", cpu_mem_size_aligned, "Bytes");
        cmd_buf = (struct dma_buffer *) malloc(sizeof(struct dma_buffer));
        cmd_buf->vaddr_user = nullptr;
        this->allocateHostDmaBuffer(cmd_buf, cpu_mem_size_aligned);

        // allocate HBM cache buffers
        this->hbm_cache_ptr = this->allocatePinGpuBuf(this->hbm_cache_size);
        this->hbm_start_offset = this->hbm_cache_ptr->phy_addr;

        // set base addresses to the driver
        struct base_addr_offsets offsets;
        offsets.dram_offsets = this->reserve_addr;
        offsets.hbm_offsets = this->hbm_start_offset;
        if (ioctl(fd_drv, IOCTL_SET_BASE_ADDR_OFFSETS, &offsets) < 0) {
            perror("ioctl: set base address offsets");
            // close();
            return;
        }

        // create AgileDmaSQHost & AgileDmaCQHost
        this->h_sq_ptrs = (AgileDmaSQHost **) malloc(dma_cmd_queue_num * sizeof(AgileDmaSQHost *));
        this->h_cq_ptrs = (AgileDmaCQHost **) malloc(dma_cmd_queue_num * sizeof(AgileDmaCQHost *));

        // create SQ & CQ for CUDA DMA
        uint64_t cpl_off = 0;
        uint64_t cmd_off = 0;
        for(uint32_t i = 0; i < this->cuda_dma_queue_num; ++i){
            this->h_sq_ptrs[i] = new AgileGpuDmaSQHost(cmd_buf->vaddr_user + cmd_off, cmd_buf->vaddr_cuda + cmd_off, this->cuda_dma_queue_depth, reserved_mem_ptr, this->hbm_cache_ptr->d_ptr, dram_start_offset, hbm_start_offset);
            this->h_cq_ptrs[i] = new AgileGpuDmaCQHost(((volatile agile_dma_cpl_t *)(this->cpl_buf->d_ptr + cpl_off)), (volatile agile_dma_cpl_t *)(this->cpl_buf->h_ptr + cpl_off), this->cuda_dma_queue_depth);
            cpl_off += this->cuda_dma_queue_depth * sizeof(struct agile_dma_cpl_t);
            cmd_off += this->cuda_dma_queue_depth * sizeof(struct agile_dma_cmd_t);
        }

        // create SQ & CQ for CPU DMA

        // tell the driver how many CPU DMA queues are created
        if(this->cpu_dma_queue_num > 0){
            if (ioctl(fd_drv, IOCTL_SET_CPU_DMA_QUEUE_NUM, &this->cpu_dma_queue_num) < 0) {
                perror("ioctl: set CPU DMA queue num");
                // close();
                return;
            }
        }

        // register CPU DMA queues to the driver
        for(uint32_t i = 0; i < this->cpu_dma_queue_num; ++i){
            
            this->h_sq_ptrs[this->cuda_dma_queue_num + i] = new AgileDmaSQHost(i, cmd_buf->vaddr_user + cmd_off, cmd_buf->vaddr_cuda + cmd_off, this->cpu_dma_queue_depth, reserved_mem_ptr, this->hbm_cache_ptr->d_ptr, dram_start_offset, hbm_start_offset);
            this->h_cq_ptrs[this->cuda_dma_queue_num + i] = new AgileDmaCQHost(((volatile agile_dma_cpl_t *)(this->cpl_buf->d_ptr + cpl_off)), (volatile agile_dma_cpl_t *)(this->cpl_buf->h_ptr + cpl_off), this->cpu_dma_queue_depth);
            struct dma_queue_pair_data queue_data;
            queue_data.queue_depth = this->cpu_dma_queue_depth;
            queue_data.cmds = this->cmd_buf->vaddr_krnl + cmd_off;
            queue_data.cpls = this->cpl_buf->buffer_params.krnl_ptr + cpl_off;
            printf("%s:%d user_ptr: %p, krnl_ptr: %p\n", __FILE__, __LINE__, cmd_buf->vaddr_user + cmd_off, this->cmd_buf->vaddr_krnl + cmd_off);
            if (ioctl(fd_drv, IOCTL_REGISTER_DMA_QUEUE_PAIRS, &queue_data) < 0) {
                perror("ioctl: create CPU DMA queue");
                return;
            }
            agile_dma_cpl_t *h_ptr = (agile_dma_cpl_t *) (this->cpl_buf->h_ptr + cpl_off);
            cpl_off += this->cpu_dma_queue_depth * sizeof(struct agile_dma_cpl_t);
            cmd_off += this->cpu_dma_queue_depth * sizeof(struct agile_dma_cmd_t);
        }

        // allocate monitors and workers
        if(this->monitor_num > this->dma_cmd_queue_num){
            this->monitor_num = this->dma_cmd_queue_num;
            INFO("reduce monitor threads to", this->monitor_num);
        }
        this->host_monitors = (AgileHostMonitor **) malloc(sizeof(AgileHostMonitor *) * this->monitor_num);
        uint32_t queues_per_monitor = this->dma_cmd_queue_num / this->monitor_num;
        for(uint32_t i = 0; i < this->monitor_num; ++i){
            
            this->host_monitors[i] = new AgileHostMonitor(i, fd_drv, dram_start_offset, hbm_start_offset, this->h_sq_ptrs + i * queues_per_monitor, (i == this->monitor_num - 1) ? (this->dma_cmd_queue_num - i * queues_per_monitor) : queues_per_monitor);
        }

        // only create workers for CUDA DMA queues
        if(this->worker_num > this->cuda_dma_queue_num){
            this->worker_num = this->cuda_dma_queue_num;
            INFO("reduce worker threads to", this->worker_num);
        }

        if(this->worker_num > 0){
            this->host_workers = (AgileHostWorker **) malloc(sizeof(AgileHostWorker *) * this->worker_num);
            queues_per_monitor = this->cuda_dma_queue_num / this->worker_num;
            for(uint32_t i = 0; i < this->worker_num; ++i){
                this->host_workers[i] = new AgileHostWorker(i, (AgileGpuDmaSQHost **)(this->h_sq_ptrs + i * queues_per_monitor), (AgileGpuDmaCQHost **)(this->h_cq_ptrs + i * queues_per_monitor), (i == this->worker_num - 1) ? (this->cuda_dma_queue_num - i * queues_per_monitor) : queues_per_monitor);
            }
        }

        // create device queue pair
        queue_pairs = (AgileDmaQueuePairDevice *) malloc(sizeof(AgileDmaQueuePairDevice) * this->dma_cmd_queue_num);
        memset(queue_pairs, 0, sizeof(AgileDmaQueuePairDevice) * this->dma_cmd_queue_num);
        for(uint32_t i = 0; i < this->dma_cmd_queue_num; ++i){
            queue_pairs[i].sq = this->h_sq_ptrs[i]->getDevicePtr();
            queue_pairs[i].cq = this->h_cq_ptrs[i]->getDevicePtr();
        }
        cuda_err_chk(cudaMalloc(&d_queue_pairs, sizeof(AgileDmaQueuePairDevice) * this->dma_cmd_queue_num));
        cuda_err_chk(cudaMemcpy(d_queue_pairs, queue_pairs, sizeof(AgileDmaQueuePairDevice) * this->dma_cmd_queue_num, cudaMemcpyHostToDevice));

        
        
    }

    AgileGpuMem getReservedMem(){
        AgileGpuMem mem;
        mem.d_ptr = cpl_buf->d_ptr + this->cpl_reserved_offset;
        mem.h_ptr = cpl_buf->h_ptr + this->cpl_reserved_offset;
        mem.size = cpl_reserved_size;
        mem.phy_addr = cpl_buf->phy_addr + this->cpl_reserved_offset;
        return mem;
    }

    uint32_t getDmaQueuePairNum(){
        return this->dma_cmd_queue_num;
    }

    void freeHost(){

        if(queue_pairs){
            free(queue_pairs);
        }
        queue_pairs = nullptr;

        // free workers and monitors
        for(uint32_t i = 0; i < this->monitor_num; ++i){
            delete this->host_monitors[i];
        }
        free(this->host_monitors);
        this->host_monitors = nullptr;

        for(uint32_t i = 0; i < this->worker_num; ++i){
            delete this->host_workers[i];
        }
        if(this->host_workers){
            free(this->host_workers);
        }
        this->host_workers = nullptr;

        // unregister cpu dma queue
        if(this->cpu_dma_queue_num > 0){
            if (ioctl(fd_drv, IOCTL_FREE_CPU_DMA_QUEUES, 0) < 0) {
                perror("ioctl: unregister CPU DMA queue");
                close();
            }
        }

        // free SQ & CQ for CUDA DMA
        for(uint32_t i = 0; i < this->cuda_dma_queue_num; ++i){
            delete this->h_cq_ptrs[i];
            delete this->h_sq_ptrs[i];
        }

        // free SQ & CQ for CPU DMA
        for(uint32_t i = 0; i < this->cpu_dma_queue_num; ++i){
            delete this->h_cq_ptrs[this->cuda_dma_queue_num + i];
            delete this->h_sq_ptrs[this->cuda_dma_queue_num + i];
        }

        if(this->h_sq_ptrs){
            free(this->h_sq_ptrs);
        }

        if(this->h_cq_ptrs){
            free(this->h_cq_ptrs);
        }

        if(this->cpl_buf){
            this->freeGpuBuf(this->cpl_buf);
        }
        
        if(cmd_buf){
            this->freeHostDmaBuffer(cmd_buf);
            free(cmd_buf);
        }

        if(this->hbm_cache_ptr){
            this->freeGpuBuf(this->hbm_cache_ptr);
        }
        this->hbm_cache_ptr = nullptr;

    }


    AgileDmaEngine * getDmaEngineDevice(){
        if(this->engine == nullptr){
            this->engine = new AgileDmaEngine(d_queue_pairs, dma_cmd_queue_num, cuda_dma_queue_num, cpu_dma_queue_num);
        }
        return this->engine->getDevicePtr();
    }

    AgileDmaQueuePairDevice * getAgileDmaQueuePairDevicePtr(){
        return this->d_queue_pairs;
    }

    void setHbmCacheSize(uint64_t size){
        this->hbm_cache_size = size;
    }

    void setDramCacheSize(uint64_t size){
        this->dram_cache_size = size;
    }

private:
    int fd_drv;
    int fd_gpu;
    int fd_mem;
    
    uint64_t reserve_size = 34359738368L;
    uint64_t reserve_addr = 0x2000000000;
    void *reserved_mem_ptr = nullptr;

    uint64_t dram_cache_size = 34359738368L;

    uint64_t hbm_cache_size = 0;
    AgileGpuMem * hbm_cache_ptr = nullptr;

    uint32_t total_dma_channels;

    AgileDmaQueuePairDevice *queue_pairs = nullptr;
    AgileDmaQueuePairDevice *d_queue_pairs = nullptr;

    AgileDmaSQHost ** h_sq_ptrs = nullptr;
    AgileDmaCQHost ** h_cq_ptrs = nullptr;

    uint32_t dma_cmd_queue_num;
    // uint32_t dma_cmd_queue_depth;

    uint32_t cuda_dma_queue_num;
    uint32_t cuda_dma_queue_depth;

    uint32_t cpu_dma_queue_num;
    uint32_t cpu_dma_queue_depth;

    AgileGpuMem * cpl_buf = nullptr;
    struct dma_buffer * cmd_buf = nullptr;
    uint64_t cpl_reserved_offset;
    uint64_t cpl_reserved_size;

    AgileHostMonitor **host_monitors = nullptr;
    uint32_t monitor_num = 1;

    AgileHostWorker **host_workers = nullptr;
    uint32_t worker_num = 1;

    AgileDmaEngine *engine = nullptr;

    std::vector<std::thread> worker_threads;

    std::vector<AgileGpuMem*> allocated_buffers;

    uint64_t dram_start_offset = 0x2000000000;
    uint64_t hbm_start_offset = 0x0;
};


