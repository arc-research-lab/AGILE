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

#include "agile_kernel_driver.h"
#include "agile_dma_service.h"
#include "agile_gpu_mem.h"

#include "io_utils.h"


#define ALIGN_UP(x, a) (((x) + ((a) - 1)) & ~((a) - 1))


typedef struct {
    uint32_t stop_sig;
} shared_hbm_t;


// typedef struct {
// } shared_dram_t;

class AgileDriver {
public:
    AgileDriver(){
        engine = nullptr;
    }
    ~AgileDriver(){}

    bool open(){
        fd = ::open("/dev/AGILE-kernel", O_RDWR);
        if (fd < 0) {
            perror("Failed to open device");
            return false;
        }
        if (ioctl(fd, IOCTL_GET_TOTAL_DMA_CHANNELS, &total_dma_channels) < 0) {
            perror("ioctl: get total DMA channels");
            close();
            return false;
        }

        fd_mem = ::open("/dev/AGILE-reserved_mem", O_RDWR);
        
        if (fd_mem < 0) {
            perror("Failed to open memory device");
            close();
            return false;
        }
        cpu_mem.ptr = ::mmap(NULL, reserve_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_mem, 0);
        if (cpu_mem.ptr == MAP_FAILED) {
            perror("mmap");
            close();
            return false;
        }
        cpu_mem.size = reserve_size;
        
        cuda_err_chk(cudaHostRegister(this->cpu_mem.ptr, reserve_size, cudaHostRegisterIoMemory));
        
        return true;
    }

    bool close(){
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }

        if (cpu_mem.ptr) {
            cuda_err_chk(cudaHostUnregister(cpu_mem.ptr));
            ::munmap(cpu_mem.ptr, cpu_mem.size);
            cpu_mem.ptr = NULL;
        }

        if (fd_mem >= 0) {
            ::close(fd_mem);
            fd_mem = -1;
        }
        return true;
    }

    managed_cpu_mem * getReserveMem() {
        return &cpu_mem;
    }

    bool allocateDmaBuffer(struct dma_buffer * buf, size_t size){
        if (!buf) {
            std::cerr << "Invalid DMA buffer." << std::endl;
            return false;
        }
        if(buf->vaddr_user){
            std::cerr << "DMA buffer is already allocated." << std::endl;
            return false;
        }
        buf->size = size;
        if (ioctl(fd, IOCTL_ALLOCATE_CACHE_BUFFER, buf) < 0) {
            perror("ioctl: allocate cache buffer");
            close();
            return false;
        }
        buf->vaddr_user = mmap(NULL, buf->size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (buf->vaddr_user == MAP_FAILED) {
            perror("mmap");
            ioctl(fd, IOCTL_FREE_CACHE_BUFFER, buf);
            close();
            return false;
        }
        return true;
    }

    bool freeDmaBuffer(struct dma_buffer * buf){
        if (buf->vaddr_user) {
            munmap(buf->vaddr_user, buf->size);
            buf->vaddr_user = NULL;
        }
        if (ioctl(fd, IOCTL_FREE_CACHE_BUFFER, buf) < 0) {
            perror("ioctl: free cache buffer");
            return false;
        }
        return true;
    }

    // bool setCplQueueNum(int num){
    //     if (num < 1) {
    //         std::cerr << "Invalid number of completion queues." << std::endl;
    //         return false;
    //     }

    //     this->num_cpl_queues = num;
    //     if (ioctl(fd, IOCTL_ALLOC_CPL_QUEUE_ARRAY, &num) < 0) {
    //         perror("ioctl: set completion queue number");
    //         return false;
    //     }
    //     return true;
    // }

    bool initSockets(){

        // set the number of CPU DMA completion queues to the kernel
        if (ioctl(fd, IOCTL_ALLOC_CPL_QUEUE_ARRAY, &this->worker_num) < 0) {
            perror("ioctl: set completion queue number");
            return false;
        }


        this->u_socket_pairs = (struct socket_pair *) malloc(this->worker_num * sizeof(struct socket_pair));
        if (!this->u_socket_pairs) {
            std::cerr << "Failed to allocate user socket pairs" << std::endl;
            return false;
        }

        this->c_socket_pairs = (struct socket_pair *) malloc(this->worker_num * sizeof(struct socket_pair));
        if (!this->c_socket_pairs) {
            std::cerr << "Failed to allocate CUDA socket pairs" << std::endl;
            return false;
        }

        this->stop_signals = (struct socket_pair *) malloc(this->worker_num * sizeof(struct socket_pair));
        if (!this->stop_signals) {
            std::cerr << "Failed to allocate stop signals" << std::endl;
            return false;
        }

        this->k_event = (int *) malloc(this->worker_num * sizeof(int));
        if (!this->k_event) {
            std::cerr << "Failed to allocate kernel events" << std::endl;
            return false;
        }

        for (int i = 0; i < this->worker_num; ++i) {

            if (socketpair(AF_UNIX, SOCK_SEQPACKET, 0, this->u_socket_pairs[i].msg_sock) < 0) {
                perror("socketpair");
                return false;
            }

            if (socketpair(AF_UNIX, SOCK_SEQPACKET, 0, this->c_socket_pairs[i].msg_sock) < 0) {
                perror("socketpair");
                return false;
            }

            if (socketpair(AF_UNIX, SOCK_STREAM, 0, this->stop_signals[i].msg_sock) < 0) {
                perror("socketpair for stop signals");
                return false;
            }

            this->k_event[i] = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
            if (this->k_event[i] < 0) {
                perror("eventfd");
                return false;
            }
        }

        // Allocate DMA buffers for completion queues
        cpl_dma_buffers = (struct dma_buffer *) malloc(this->worker_num * sizeof(struct dma_buffer));
        memset(cpl_dma_buffers, 0, this->worker_num * sizeof(struct dma_buffer));
        for (int i = 0; i < this->worker_num; ++i) {
            if(!this->allocateDmaBuffer(&cpl_dma_buffers[i], 4096)){
                std::cerr << "Failed to allocate DMA buffer for completion queue " << i << std::endl;
                return false;
            }
        }

        // Create completion queues for user space application
        cpl_queues = (struct dma_cpl_queue *) malloc(this->worker_num * sizeof(struct dma_cpl_queue));
        for (int i = 0; i < this->worker_num; ++i) {
            cpl_queues[i].data.ptr = cpl_dma_buffers[i].vaddr_user;
            cpl_queues[i].head = 0;
            cpl_queues[i].tail = 0;
            cpl_queues[i].idx = i;
            cpl_queues[i].depth = cpl_dma_buffers[i].size / sizeof(struct dma_cpl_entry);
            memset(cpl_queues[i].data.ptr, 0, cpl_dma_buffers[i].size);
        }

        // Register the cpl_dma_buffer to kernel space
        for (int i = 0; i < this->worker_num; ++i) {
            struct cpl_queue_config cpl_cfg;
            cpl_cfg.size = cpl_dma_buffers[i].size;
            cpl_cfg.depth = cpl_dma_buffers[i].size / sizeof(struct dma_cpl_entry);
            cpl_cfg.vaddr_krnl = cpl_dma_buffers[i].vaddr_krnl;
            cpl_cfg.idx = i;
            cpl_cfg.efd = this->k_event[i];
            if (ioctl(fd, IOCTL_SET_CPL_QUEUE, &cpl_cfg) < 0) {
                perror("ioctl: register completion queue buffer");
                return false;
            }
        }

        // Create the completion queues array
        cpl_queues_info = (struct dma_cpl_queues_info *) malloc(sizeof(struct dma_cpl_queues_info));
        if (!cpl_queues_info) {
            std::cerr << "Failed to allocate DMA completion queues info" << std::endl;
            return false;
        }
        cpl_queues_info->queues = cpl_queues;
        cpl_queues_info->num_queues = num_cpl_queues;

        this->dma_worker_idx = 0;

        return true;
    }

    void stopWorkers(){
        for(int i = 0; i < this->worker_num; ++i){
            uint64_t val = 1;
            write(this->stop_signals[i].msg_sock[1], &val, sizeof(val));
        }
    }

    bool freeCplQueues() {

        for(int i = 0; i < this->worker_num; ++i){
            ::close(this->u_socket_pairs[i].msg_sock[SOCKET_SEND]);
            ::close(this->u_socket_pairs[i].msg_sock[SOCKET_RECEIVE]);
            ::close(this->c_socket_pairs[i].msg_sock[SOCKET_SEND]);
            ::close(this->c_socket_pairs[i].msg_sock[SOCKET_RECEIVE]);
            ::close(this->stop_signals[i].msg_sock[SOCKET_SEND]);
            ::close(this->stop_signals[i].msg_sock[SOCKET_RECEIVE]);
            ::close(this->k_event[i]);
            // ::close(this->efd[i]);
        }

        delete this->dispatcher;

        // Unregister completion queues with the kernel
        for(int i = 0; i < this->worker_num; ++i){
        
            struct cpl_queue_config cpl_cfg;
            cpl_cfg.size = cpl_dma_buffers[i].size;
            cpl_cfg.depth = cpl_dma_buffers[i].size / sizeof(struct dma_cpl_entry);
            cpl_cfg.vaddr_krnl = cpl_dma_buffers[i].vaddr_krnl;
            cpl_cfg.idx = i;
            if (ioctl(fd, IOCTL_DEL_CPL_QUEUE, &cpl_cfg) < 0) {
                perror("ioctl: unregister completion queue");
            }
            this->freeDmaBuffer(&cpl_dma_buffers[i]);
        }
        // Free completion queue array with the kernel
        if(ioctl(fd, IOCTL_FREE_CPL_QUEUE_ARRAY, NULL) < 0) {
            perror("ioctl: free completion queue array");
            return false;
        }

        if (cpl_queues_info) {
            free(cpl_queues_info);
            cpl_queues_info = NULL;
        }

        free(cpl_queues);
        free(cpl_dma_buffers);

        free(this->u_socket_pairs);
        free(this->c_socket_pairs);
        free(this->stop_signals);
        free(this->k_event);

        return true;
    }

    void setPinedGpuMem(pined_gpu_mem *pined_mem){
        this->pined_mem = *pined_mem;
    }

    void setDmaCmdQueueNum(uint32_t dma_cmd_queue_num){
        this->dma_cmd_queue_num = dma_cmd_queue_num;
    }

    void setDmaCmdQueueDepth(uint32_t dma_cmd_queue_depth){
        this->dma_cmd_queue_depth = dma_cmd_queue_depth;
    }

    void setMonitorNum(uint32_t monitor_num){
        this->monitor_num = monitor_num;
    }

    void setWorkerNum(uint32_t worker_num){
        this->worker_num = worker_num;
    }

    void allocateHost(){
        if(this->dma_cmd_queue_num == 0 || this->dma_cmd_queue_depth == 0 || this->monitor_num == 0 || this->worker_num == 0){
            std::cerr << "host config error\n";
            return;
        }
        this->allocateHost(dma_cmd_queue_num, dma_cmd_queue_depth, monitor_num, worker_num);
    }

    void allocateHost(uint32_t dma_cmd_queue_num, uint32_t dma_cmd_queue_depth, uint32_t monitor_num, uint32_t worker_num){

        if(monitor_num > dma_cmd_queue_num){
            monitor_num = dma_cmd_queue_num;
        }
        
        this->dma_cmd_queue_num = dma_cmd_queue_num;
        this->dma_cmd_queue_depth = dma_cmd_queue_depth;
        this->queue_pairs = (AgileDmaQueuePairHost *) malloc(dma_cmd_queue_num * sizeof(AgileDmaQueuePairHost));
        this->worker_num = worker_num;
        this->monitor_num = monitor_num;


        uint32_t queues_per_monitor = dma_cmd_queue_num / monitor_num;
        cuda_err_chk(cudaMalloc(&this->d_queue_pairs, dma_cmd_queue_num * sizeof(AgileDmaQueuePairDevice)));
        for(uint32_t i = 0; i < dma_cmd_queue_num; ++i){
            AgileDmaQueuePairDevice d_pair;
            this->queue_pairs[i].sq = new AgileDmaSQHost(dma_cmd_queue_depth);
            this->queue_pairs[i].cq = new AgileDmaCQHost(((void*)this->pined_mem.d_ptr) + i * dma_cmd_queue_depth * sizeof(dma_cpl_t), this->pined_mem.h_ptr + i * dma_cmd_queue_depth * sizeof(dma_cpl_t), dma_cmd_queue_depth);
            d_pair.sq = this->queue_pairs[i].sq->getDevicePtr();
            d_pair.cq = this->queue_pairs[i].cq->getDevicePtr();
            cuda_err_chk(cudaMemcpy(this->d_queue_pairs + i, &d_pair, sizeof(AgileDmaQueuePairDevice), cudaMemcpyHostToDevice));   
        }
        // this->setCplQueueNum(dma_cmd_queue_num);
        this->initSockets();
        // memset(this->cpu_mem.ptr, 0, this->getCpuMemOffset());
        void * host_ptr = (void *) this->cpu_mem.ptr; // + this->getCpuMemOffset();
        void * device_ptr = (void *) this->pined_mem.d_ptr + this->getPinedMemOffset();
        this->host_workers = new AgileHostWorker*[worker_num];
        for(uint32_t i = 0; i < worker_num; ++i){
            host_workers[i] = new AgileHostWorker(i, k_event[i], u_socket_pairs[i].msg_sock[SOCKET_RECEIVE], c_socket_pairs[i].msg_sock[SOCKET_RECEIVE], stop_signals[i].msg_sock[SOCKET_RECEIVE], &cpl_queues[i], &this->queue_pairs[i * queues_per_monitor], (i == worker_num - 1) ? (dma_cmd_queue_num - (worker_num - 1) * queues_per_monitor) : queues_per_monitor);
        }
        this->dispatcher = new AgileDispatcher(this->host_workers, this->u_socket_pairs, this->c_socket_pairs, this->worker_num);
        this->host_monitors = new AgileHostMonitor*[monitor_num];
        for(uint32_t i = 0; i < monitor_num; ++i){
            host_monitors[i] = new AgileHostMonitor(i, this->fd, this->reserve_addr + this->getCpuMemOffset(), this->pined_mem.addr + this->getPinedMemOffset(), &this->queue_pairs[i * queues_per_monitor], (i == monitor_num - 1) ? (dma_cmd_queue_num - (monitor_num - 1) * queues_per_monitor) : queues_per_monitor, this->dispatcher);
            host_monitors[i]->setHostPtr(host_ptr); // (this->managed_ptr);
            host_monitors[i]->setDevicePtr(device_ptr);
        }
    }

    void startWorker(){
        for(uint32_t i = 0; i < worker_num; ++i){
            worker_threads.emplace_back([this, i](){
                // pin_current_thread_to(i*2 + 8);
                int cpu_id = sched_getcpu();
                int node = numa_node_of_cpu(cpu_id);
                INFO("start working on CPU", cpu_id, "NUMA node", node);
                // this->host_workers[i]->waitEvents();
                for(;;){
                    if(-1 == this->host_workers[i]->waitEvents()){
                        break;
                    }
                    if(-1 == this->host_workers[i]->processEvents()){
                        break;
                    }
                }
            });
        }
    }

    void stopWorker(){
        for(uint32_t i = 0; i < worker_num; ++i){
            // write some data to stop_signals
            int sig = 1;
            send(this->stop_signals[i].msg_sock[SOCKET_SEND], &sig, sizeof(sig), 0);
            this->host_workers[i]->stop();
        }

        for(uint32_t i = 0; i < worker_num; ++i){
            worker_threads[i].join();
        }
    }

    void startMonitor(){
        for(uint32_t i = 0; i < monitor_num; ++i){
            host_monitors[i]->start();
        }
    }

    void stopMonitor(){
        for(uint32_t i = 0; i < monitor_num; ++i){
            host_monitors[i]->stop();
        }
    }

    void * getHostMemPtr(){
        return (void *) this->cpu_mem.ptr; // + this->getCpuMemOffset();
    }

    void * getDeviceMemPtr(){
        return (void *) this->pined_mem.d_ptr + this->getPinedMemOffset();
    }

    void * getDeviceMemMappedPtr(){
        return (void *) this->pined_mem.h_ptr + this->getPinedMemOffset();
    }

    uint64_t getCpuMemOffset(){
        return ALIGN_UP(((uint64_t)dma_cmd_queue_num) * sizeof(dma_cmd_t) * ((uint64_t)dma_cmd_queue_depth), 65536L);
    }
    
    uint64_t getPinedMemOffset(){
        return ALIGN_UP(((uint64_t)dma_cmd_queue_num) * sizeof(dma_cpl_t) * ((uint64_t)dma_cmd_queue_depth) + sizeof(shared_hbm_t), 65536L);
    }

    shared_hbm_t * getSharedHbmHostPtr(){
        return (shared_hbm_t *) this->pined_mem.h_ptr + ((uint64_t)dma_cmd_queue_num) * sizeof(dma_cpl_t) * ((uint64_t)dma_cmd_queue_depth);
    }

    shared_hbm_t * getSharedHbmDevicePtr(){
        return (shared_hbm_t *) ((void*)this->pined_mem.d_ptr) + ((uint64_t)dma_cmd_queue_num) * sizeof(dma_cpl_t) * ((uint64_t)dma_cmd_queue_depth);
    }

    AgileHostMonitor ** getHostMonitors(){
        return this->host_monitors;
    }

    AgileHostWorker ** getHostWorkers(){
        return this->host_workers;
    }

    void freeHost(){
        // free host monitor
        this->freeCplQueues();
        for(uint32_t i = 0; i < monitor_num; ++i){
            delete this->host_monitors[i];
        }
        delete[] this->host_monitors;
        // free host worker
        for(uint32_t i = 0; i < worker_num; ++i){
            delete this->host_workers[i];
        }
        delete[] this->host_workers;
        for(uint32_t i = 0; i < dma_cmd_queue_num; ++i){
            delete this->queue_pairs[i].cq;
            delete this->queue_pairs[i].sq;
        }
        free(this->queue_pairs);
        if(this->engine){
            delete engine;
        }
    }

    AgileDmaEngine * getDmaEngine(){
        if(this->engine == nullptr){
            this->engine = new AgileDmaEngine(d_queue_pairs, dma_cmd_queue_num);
        }
        return this->engine;
    }

    AgileDmaQueuePairDevice * getAgileDmaQueuePairDevicePtr(){
        return this->d_queue_pairs;
    }

    void setManagedPtr(void * ptr){
        this->managed_ptr = ptr;
    }

private:
    int fd;
    int fd_mem;
    
    // void * mem_reserve;
    uint64_t reserve_size = 34359738368L;
    uint64_t reserve_addr = 0x2000000000;
    managed_cpu_mem cpu_mem;
    pined_gpu_mem pined_mem;
    void * managed_ptr;

    int num_cpl_queues;
    int dma_worker_idx;
    uint32_t total_dma_channels;
    struct dma_cpl_queue *cpl_queues; // each thread has its own completion queue, where the coroutine handles are stored
    struct dma_buffer *cpl_dma_buffers;
    struct dma_cpl_queues_info *cpl_queues_info; // holds the information about the completion queues
    struct socket_pair *u_socket_pairs; // generate new coroutines
    struct socket_pair *c_socket_pairs; // cuda asynchrnous memory copy callback 
    struct socket_pair *stop_signals;
    int *k_event;
    // int *efd;

    AgileDmaQueuePairHost *queue_pairs;
    AgileDmaQueuePairDevice *d_queue_pairs;
    uint32_t dma_cmd_queue_num;
    uint32_t dma_cmd_queue_depth;

    AgileHostMonitor **host_monitors;
    uint32_t monitor_num;

    AgileHostWorker **host_workers;
    uint32_t worker_num;

    AgileDispatcher *dispatcher;

    AgileDmaEngine *engine;

    std::vector<std::thread> worker_threads;
};


