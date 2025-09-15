#pragma once
#include <atomic>
#include <immintrin.h>

#include "agile_dma_cmd.h"
#include "agile_lock.h"
#include "io_utils.h"


static inline uint64_t ts_to_ns(const struct timespec* ts) {
    return (uint64_t)ts->tv_sec * 1000000000ull + (uint64_t)ts->tv_nsec;
}


__device__
bool sq_entry_try_lock(uint32_t * lock) {
    uint32_t val = -1;
    if((val = atomicCAS(lock, SQ_CMD_EMPTY, SQ_CMD_OCCUPY)) != SQ_CMD_EMPTY){
        // fail to acquire the lock
        return false;
    }
    return true;
}

__device__
bool sq_entry_finish_upd(uint32_t * lock){
    uint32_t val = -1;
    if((val = atomicCAS(lock, SQ_CMD_OCCUPY, SQ_CMD_FINISHED)) != SQ_CMD_OCCUPY){
        printf("fail to update the lock %d\n", val);
        return false;
    }
    return true;
}

__device__
bool wait_cmd(uint32_t * lock){
    uint32_t val = -1;
    while((val = atomicCAS(lock, SQ_CMD_FINISHED, SQ_CMD_EMPTY)) != SQ_CMD_FINISHED){
    }
    return true;
}

template<class T>
class DevicePtr {
    T * d_ptr = nullptr;
public:
    __host__
    T * getDevicePtr(){
        if(d_ptr == nullptr){
            cuda_err_chk(cudaMalloc(&d_ptr, sizeof(T)));
            cuda_err_chk(cudaMemcpy(d_ptr, static_cast<T*>(this), sizeof(T), cudaMemcpyHostToDevice));
        }
        return d_ptr;
    }
};

class AgileDmaCQDevice {
public:
    volatile struct agile_dma_cpl_t *cpl;
    unsigned int depth;
    unsigned int head_offset;
    unsigned int mask;

    __host__
    AgileDmaCQDevice(volatile struct agile_dma_cpl_t *cpl, unsigned int depth) : cpl(cpl), depth(depth), head_offset(0), mask(0) {}
};

class AgileDmaCQHost {
public:
    volatile struct agile_dma_cpl_t *d_cpl;
    volatile struct agile_dma_cpl_t *h_cpl;
    unsigned int depth;
    AgileDmaCQDevice * ptr = nullptr;

    __host__
    AgileDmaCQHost(volatile struct agile_dma_cpl_t *d_cpl, volatile struct agile_dma_cpl_t *h_cpl, unsigned int depth) : d_cpl(d_cpl), h_cpl(h_cpl), depth(depth) {
    }  

    __host__
    AgileDmaCQDevice * getDevicePtr(){
        if(this->ptr == nullptr){
            auto h_ptr = new AgileDmaCQDevice(this->d_cpl, this->depth);
            cuda_err_chk(cudaMalloc(&this->ptr, sizeof(AgileDmaCQDevice)));
            cuda_err_chk(cudaMemcpy(this->ptr, h_ptr, sizeof(AgileDmaCQDevice), cudaMemcpyHostToDevice));
            delete h_ptr;
        }
        return this->ptr;
    }
};

// agile_dma_cpl_t is located on the HBM and exposed to user space and kernel space
class AgileGpuDmaCQHost : public AgileDmaCQHost {
public:
    
    std::atomic_uint g_tail{0}; // used in the host driver
    
    __host__
    AgileGpuDmaCQHost(volatile struct agile_dma_cpl_t *d_cpl, volatile struct agile_dma_cpl_t *h_cpl, unsigned int depth) : AgileDmaCQHost(d_cpl, h_cpl, depth) {
    }

    __host__
    ~AgileGpuDmaCQHost() {
    }

    __host__
    void updateCpl(uint32_t identifier) {
        unsigned int cpl_idx = g_tail.fetch_add(1) % depth;
        while(h_cpl[cpl_idx].status != DMA_CPL_EMPTY){
        }
        h_cpl[cpl_idx].identifier = identifier;
        _mm_sfence();
        h_cpl[cpl_idx].status = DMA_CPL_READY;
    }
};


class AgileDmaSQDevice {
    __host__
    AgileDmaSQDevice(){}

public:
    volatile agile_dma_cmd_t * cmd;
    unsigned int depth;
    unsigned int tail;
    uint32_t * cmd_status; // empty; occupied; finished;
    uint32_t dma_engine_type; // GPU_DMA_ENGINE; CPU_DMA_ENGINE; used as a hint if anyone want to prioritize certain DMA engine
    uint32_t dma_engine_idx; // another hint

    __host__
    AgileDmaSQDevice(volatile agile_dma_cmd_t * cmd, uint32_t * cmd_status, unsigned int depth) : cmd(cmd), cmd_status(cmd_status), depth(depth), tail(0) {
    }

    __host__
    ~AgileDmaSQDevice() {
    }

    __device__
    bool submitDMA_try(uint64_t src_offset, uint64_t dst_offset, unsigned int size, char direction, uint32_t **lock, AgileLockChain * chain) {
        unsigned int cmd_pos_g = atomicAdd(&this->tail, 0);
        unsigned int cmd_pos = cmd_pos_g % this->depth;
        
        if(sq_entry_try_lock(&cmd_status[cmd_pos])){
            atomicAdd(&this->tail, 1);
            cmd[cmd_pos].src_offset = src_offset;
            cmd[cmd_pos].dst_offset = dst_offset;
            cmd[cmd_pos].size = size;
            cmd[cmd_pos].direction = direction;
            __threadfence_system();
            cmd[cmd_pos].trigger = SQ_CMD_OCCUPY;
            *lock = &cmd_status[cmd_pos];
            return true;
        // }else{
        //     printf("fail to submit cmd at pos %d trigger %d\n", cmd_pos, cmd[cmd_pos].trigger);
        }
        return false;
    }
};

/**
 * If CPU DMA engines are used, the monitor threads will poll the commands, do batching, asign to certain DMA engine, and issue DMA transfers via ioctl.
 * There is no worker threads to check the completion of the DMA transfer, and the DMA callback function will directly update the CPL to GPU in the kernel space.
 * 
 * If GPU DMA engine is used (extend to class AgileGpuDmaSQHost), the monitor will poll the commands and issue DMA transfer via cudaMemcpyAsync. 
 * It requires worker threads to poll the status of cudaEvent_t to see if the DMA transfer is finished and then update CPL to GPU in the user space.
 */
class AgileDmaSQHost {
public:
    uint32_t k_queue_id;
    volatile struct agile_dma_cmd_t * h_cmd;
    volatile struct agile_dma_cmd_t * d_cmd;
    unsigned int depth;
    unsigned int tail;
    unsigned int head;
    AgileDmaSQDevice * ptr;
    uint32_t * d_cmd_locks;

    void * h_mem;
    void * d_mem;
    uint64_t dram_offset;
    uint64_t hbm_offset;

    uint32_t cmd_count = 0;

    AgileDmaSQHost(uint32_t k_queue_id, void * h_cmd_ptr, void * d_cmd_ptr, unsigned int depth, void * h_mem, void * d_mem, uint64_t dram_offset, uint64_t hbm_offset) \
    : k_queue_id(k_queue_id), h_cmd(static_cast<volatile struct agile_dma_cmd_t *>(h_cmd_ptr)), d_cmd(static_cast<volatile struct agile_dma_cmd_t *>(d_cmd_ptr)), \
    ptr(nullptr), tail(0), head(0), depth(depth), h_mem(h_mem), d_mem(d_mem), dram_offset(dram_offset), hbm_offset(hbm_offset) {
        cuda_err_chk(cudaMalloc(&this->d_cmd_locks, sizeof(uint32_t) * depth));
    }

    ~AgileDmaSQHost(){
        cuda_err_chk(cudaFree(this->d_cmd_locks));
    }

    /**
     * the poll function will issue DMA commands and returns how many commands are issued.
     */
    __host__
    virtual uint32_t poll(int fd, uint32_t monitor_id) {
        uint32_t count = 0;
        while(this->h_cmd[tail].trigger == SQ_CMD_OCCUPY && count < depth){
            // printf("%s:%d cmd received @ %p %d src: %llx, dst: %llx\n", __FILE__, __LINE__, &this->h_cmd[tail], tail, this->h_cmd[tail].src_offset, this->h_cmd[tail].dst_offset);
            this->h_cmd[tail].dma_engine_id = monitor_id; // cmd_count++ % 2;
            this->h_cmd[tail].trigger = SQ_CMD_ISSUED;
            count++;
            tail = (tail + 1) % depth;
            // assign to certain CPU DMA engine
        }
        if(count > 0){
            struct dma_command cmd;
            cmd.queue_id = this->k_queue_id;
            cmd.count = count;
            // printf("%s:%d submit %d cmds for queue %d\n", __FILE__, __LINE__, count, this->k_queue_id);
            if(ioctl(fd, IOCTL_SUBMIT_DMA_CMD, &cmd)){
                ERROR("submit command error");
            }
        }

        return count;
    }

    __host__
    void getBasePtrAddr(uint8_t direction, void **src, void **dst, uint64_t *src_addr, uint64_t *dst_addr){
        if(direction == DMA_CPU2GPU){
            *src = h_mem;
            *dst = d_mem;
            *src_addr = dram_offset;
            *dst_addr = hbm_offset;
        } else if (direction == DMA_GPU2CPU){
            *src = d_mem;
            *dst = h_mem;
            *dst_addr = dram_offset;
            *src_addr = hbm_offset;
        } else {
            ERROR("unknow direction");
        }
    }

    /**
     * The GPU will only know the backend DMA engine via AgileDmaSQDevice::dma_engine_type;
     */
    __host__
    AgileDmaSQDevice * getDevicePtr(){
        if(this->ptr == nullptr){
            AgileDmaSQDevice * h_ptr = new AgileDmaSQDevice(d_cmd, d_cmd_locks, depth);
            cuda_err_chk(cudaMalloc(&this->ptr, sizeof(AgileDmaSQDevice)));
            cuda_err_chk(cudaMemcpy(this->ptr, h_ptr, sizeof(AgileDmaSQDevice), cudaMemcpyHostToDevice));
            delete h_ptr;
        }
        return this->ptr;
    }

};

/**
 * If a SQ uses the GPU DMA engine, use this class instead of AgileDmaSQHost
 */
class AgileGpuDmaSQHost : public AgileDmaSQHost {
public:
    
    cudaStream_t * stream;
    cudaEvent_t * event;

    __host__
    AgileGpuDmaSQHost(void * h_cmd_ptr, void * d_cmd_ptr, unsigned int depth, void * h_mem, void * d_mem, uint64_t dram_offset, uint64_t hbm_offset) : AgileDmaSQHost(-1, h_cmd_ptr, d_cmd_ptr, depth, h_mem, d_mem, dram_offset, hbm_offset) {
        this->stream = (cudaStream_t *) malloc(sizeof(cudaStream_t) * depth);
        this->event = (cudaEvent_t *) malloc(sizeof(cudaEvent_t) * depth);
        

        for(unsigned int i = 0; i < depth; ++i){
            cuda_err_chk(cudaStreamCreateWithFlags(&this->stream[i], cudaStreamNonBlocking));
            cuda_err_chk(cudaEventCreateWithFlags(&this->event[i], cudaEventDisableTiming));
        }
    }

    __host__
    ~AgileGpuDmaSQHost() {
        
        for(unsigned int i = 0; i < depth; ++i){
            cuda_err_chk(cudaStreamDestroy(this->stream[i]));
            cuda_err_chk(cudaEventDestroy(this->event[i]));
        }
        free(this->stream);
        free(this->event);
        
    }

    __host__
    uint32_t poll(int do_not_use, uint32_t monitor_id) override {
        // printf("%s:%d poll AgileGpuDmaSQHost\n", __FILE__, __LINE__);
        if(this->h_cmd[tail].trigger == SQ_CMD_OCCUPY){
            // printf("%s:%d cmd received @ %p %d\n", __FILE__, __LINE__, &this->h_cmd[tail], tail);
            void * src;
            void * dst;
            uint64_t src_addr = 0;
            uint64_t dst_addr = 0;
            this->getBasePtrAddr(this->h_cmd[tail].direction, &src, &dst, &src_addr, &dst_addr);
            src += this->h_cmd[tail].src_offset;
            dst += this->h_cmd[tail].dst_offset;
            cuda_err_chk(cudaMemcpyAsync(dst, src, this->h_cmd[tail].size, \
                this->h_cmd[tail].direction == DMA_CPU2GPU ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost, this->stream[tail]));
            cuda_err_chk(cudaEventRecord(this->event[tail], this->stream[tail]));
            this->h_cmd[tail].trigger = SQ_CMD_ISSUED;
            tail = (tail + 1) % depth;
        }
        return 0;
    }

    __host__
    bool pollCudaEvents(uint32_t *identifier){
        // printf("%p %d\n", this, head);
        if(this->h_cmd[head].trigger == SQ_CMD_ISSUED){
            if(cudaEventQuery(this->event[head]) == cudaSuccess){
                // printf("%s:%d cmd finished @ %p %d\n", __FILE__, __LINE__, this, head);
                *identifier = head;
                this->h_cmd[head].trigger = SQ_CMD_FINISHED;
                head = (head + 1) % depth;   
                return true;
            }
        }
        return false;
    }

};

class AgileDmaQueuePairDevice {
public:
    AgileDmaSQDevice *sq;
    AgileDmaCQDevice *cq;
};

class AgileDmaEngine : public DevicePtr<AgileDmaEngine>{
public:
    AgileDmaQueuePairDevice *queue_pair;
    unsigned int queue_num;

    unsigned int cuda_dma_queue;
    unsigned int cpu_dma_queue;
    
public:
    
    __host__
    AgileDmaEngine(AgileDmaQueuePairDevice *queue_pair, unsigned int queue_num, unsigned int cuda_dma_queue, unsigned int cpu_dma_queue) : queue_pair(queue_pair), queue_num(queue_num), cuda_dma_queue(cuda_dma_queue), cpu_dma_queue(cpu_dma_queue){}
    
    __device__
    void submit(uint64_t src_offset, uint64_t dst_offset, unsigned int size, char flags, uint32_t **lock, AgileLockChain *chain) {
        uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int q_idx = tid % queue_num;
        do {
            if(queue_pair[q_idx].sq->submitDMA_try(src_offset, dst_offset, size, flags, lock, chain)){
                return;
            }
            q_idx = (q_idx + 1) % queue_num;
        } while (true);
    }
};
