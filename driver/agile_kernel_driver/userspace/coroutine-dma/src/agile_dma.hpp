#pragma once
#include <atomic>
#include <immintrin.h>

#include "agile_coroutine.h"
#include "agile_lock.h"
#include "io_utils.h"

#define DMA_CPU2GPU 0
#define DMA_GPU2CPU 1

#define DMA_CPL_EMPTY 0
#define DMA_CPL_READY 1

typedef struct {
    uint16_t identifier;
    char status;
    char phase;
} dma_cpl_t;

typedef struct {
    uint32_t queue_id;
    uint32_t cmd_id;
} cmd_info_t;

template<class T>
class DevicePtr {
public:
    __host__
    T * getDevicePtr(){
        T* ptr;
        cuda_err_chk(cudaMalloc(&ptr, sizeof(T)));
        cuda_err_chk(cudaMemcpy(ptr, static_cast<T*>(this), sizeof(T), cudaMemcpyHostToDevice));
        return ptr;
    }
};

class AgileDmaCQDevice {
public:
    volatile dma_cpl_t *cpl;
    unsigned int depth;
    unsigned int head_offset;
    unsigned int mask;

    __host__
    AgileDmaCQDevice(volatile dma_cpl_t *cpl, unsigned int depth) : cpl(cpl), depth(depth), head_offset(0), mask(0) {}
};

// Located on the HBM
class AgileDmaCQHost {
public:
    volatile dma_cpl_t *cpl;
    volatile dma_cpl_t *h_cpl;
    unsigned int depth;
    unsigned int head; // used in the CUDA kernel
    volatile unsigned int * r_head; // update the head pointer on CPU

    std::atomic_uint g_tail{0}; // used in the host driver

    AgileDmaCQDevice * ptr = nullptr;

    __host__
    AgileDmaCQHost(void * d_ptr, void * h_ptr, unsigned int depth) : cpl(static_cast<volatile dma_cpl_t *>(d_ptr)), h_cpl(static_cast<volatile dma_cpl_t *>(h_ptr)), depth(depth), head(0), r_head(nullptr) {
    }

    __host__
    ~AgileDmaCQHost() {
    }

    __host__
    void updateCpl(uint32_t identifier) {
        // cpl[identifier].status = DMA_CPL_SUCCESS;
        unsigned int cpl_idx = g_tail.fetch_add(1) % depth;
        while(h_cpl[cpl_idx].status != DMA_CPL_EMPTY){
        }
        h_cpl[cpl_idx].identifier = identifier;
        _mm_sfence();
        h_cpl[cpl_idx].status = DMA_CPL_READY;
    }

    __host__
    AgileDmaCQDevice * getDevicePtr(){
        if(this->ptr == nullptr){
            auto h_ptr = new AgileDmaCQDevice(this->cpl, this->depth);
            cuda_err_chk(cudaMalloc(&this->ptr, sizeof(AgileDmaCQDevice)));
            cuda_err_chk(cudaMemcpy(this->ptr, h_ptr, sizeof(AgileDmaCQDevice), cudaMemcpyHostToDevice));
            delete h_ptr;
        }
        return this->ptr;
    }
};


typedef struct {
    uint64_t src_addr;
    uint64_t dst_addr;
    uint32_t size;
    uint32_t identifier; // TODO: check bitwidth
    uint32_t flags; // define the direction of the DMA
    uint32_t trigger; // trigger DMA event
} dma_cmd_t; 

#define SQ_CMD_EMPTY 0
#define SQ_CMD_OCCUPY 1
#define SQ_CMD_FINISHED 2

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

class AgileDmaSQDevice {

    __host__
    AgileDmaSQDevice(){}

public:
    volatile dma_cmd_t * cmd;
    unsigned int depth;
    unsigned int tail;
    uint32_t * cmd_locks;

    __host__
    AgileDmaSQDevice(volatile dma_cmd_t * cmd, uint32_t * cmd_locks, unsigned int depth) : cmd(cmd), cmd_locks(cmd_locks), depth(depth), tail(0) {
    }

    __host__
    ~AgileDmaSQDevice() {
    }

    __device__
    bool submitDMA_try(uint64_t src_addr, uint64_t dst_addr, unsigned int size, char flags, cmd_info_t &cmd_info, AgileLockChain * chain) {
        // printf("try %p\n", this);
        unsigned int cmd_pos_g = atomicAdd(&this->tail, 0);
        // printf("try\n");
        unsigned int cmd_pos = cmd_pos_g % this->depth;
        if(sq_entry_try_lock(&cmd_locks[cmd_pos])){
            atomicAdd(&this->tail, 1);
            cmd[cmd_pos].src_addr = src_addr;
            cmd[cmd_pos].dst_addr = dst_addr;
            cmd[cmd_pos].size = size;
            cmd[cmd_pos].flags = flags;
            cmd[cmd_pos].identifier = cmd_pos;
            __threadfence_system();
            cmd[cmd_pos].trigger = 1;
            cmd_info.cmd_id = cmd_pos;
           
            return true;
        }else{
            // printf("fail to submit cmd at pos %d trigger %d\n", cmd_pos, cmd[cmd_pos].trigger);
        }
        return false;
    }
};

typedef struct {
  uint64_t coroutine_idx;
    DmaRequest::promise_type *promise;
    AgileDmaCQHost *cq;
    uint32_t identifier;
} DmaCallbackArgs;

static void CUDART_CB host_cb(void* p) {
    DmaCallbackArgs* args = static_cast<DmaCallbackArgs*>(p);
    // args->promise->t5 = std::chrono::high_resolution_clock::now();
    args->cq->updateCpl(args->identifier);
    // args->promise->t6 = std::chrono::high_resolution_clock::now();

    // Calculate and log the time taken for each stage
    // auto time0 = args->promise->t1 - args->promise->t0;
    // auto time1 = args->promise->t2 - args->promise->t1;
    // auto time2 = args->promise->t3 - args->promise->t2;
    // auto time3 = args->promise->t4 - args->promise->t3;
    // auto time4 = args->promise->t5 - args->promise->t4;
    // auto time5 = args->promise->t6 - args->promise->t5;

    // printf("%lld, %lld, %lld, %lld, %lld, %lld\n",
    //        time0.count(), time1.count(), time2.count(), time3.count(),
    //        time4.count(), time5.count());
}

typedef struct{
    std::chrono::high_resolution_clock::time_point t0;
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::high_resolution_clock::time_point t2;
    std::chrono::high_resolution_clock::time_point t3;
    std::chrono::high_resolution_clock::time_point t4;
    std::chrono::high_resolution_clock::time_point t5;
    std::chrono::high_resolution_clock::time_point t6;
} monitor_log_item;

typedef struct {
    uint64_t coroutine_idx;
    int cb_socket_sender;
    DmaRequest::promise_type *promise;
    AgileDmaCQHost *cq;
    uint32_t identifier;
    monitor_log_item * log_item;
} cuda_callback_param;

// Located on the DRAM  
class AgileDmaSQHost {
public:
    volatile dma_cmd_t * cmd;
    unsigned int depth;
    unsigned int tail;
    unsigned int head;
    // void * h_cmd;
    
    cudaStream_t * stream;
    cudaEvent_t * event;
    cuda_callback_param *cb_param;

    // cudaGraph_t * graph;
    // cudaGraphNode_t * memcpyNode;
    // cudaGraphNode_t * callbackNode;
    // cudaGraphExec_t * graphExec;
    // DmaCallbackArgs * cb_args;

    AgileDmaSQDevice * ptr;
    uint32_t * d_cmd_locks;


    // void* dummy_h = nullptr;   // pinned host
    // void* dummy_d = nullptr;   // device

    __host__
    AgileDmaSQHost(unsigned int depth) : depth(depth), tail(0), head(0), ptr(nullptr) {
        this->stream = (cudaStream_t *) malloc(sizeof(cudaStream_t) * depth);
        this->event = (cudaEvent_t *) malloc(sizeof(cudaEvent_t) * depth);
        // this->graph = (cudaGraph_t *) malloc(sizeof(cudaGraph_t) * depth);
        // this->graphExec = (cudaGraphExec_t *) malloc(sizeof(cudaGraphExec_t) * depth);
        // this->memcpyNode = (cudaGraphNode_t *) malloc(sizeof(cudaGraphNode_t) * depth);
        // this->callbackNode = (cudaGraphNode_t *) malloc(sizeof(cudaGraphNode_t) * depth);
        // this->cb_args = (DmaCallbackArgs *) malloc(sizeof(DmaCallbackArgs) * depth);
        this->cb_param = (cuda_callback_param *) malloc(sizeof(cuda_callback_param) * depth);
        cuda_err_chk(cudaMalloc(&this->d_cmd_locks, sizeof(uint32_t) * depth));

        // cudaHostAlloc(&dummy_h, 1, cudaHostAllocPortable); // pinned
        // cudaMalloc(&dummy_d, 1);

        for(unsigned int i = 0; i < depth; ++i){
            cuda_err_chk(cudaStreamCreateWithFlags(&this->stream[i], cudaStreamNonBlocking));
            cuda_err_chk(cudaEventCreateWithFlags(&this->event[i], cudaEventDisableTiming));

            // cuda_err_chk(cudaGraphCreate(&this->graph[i], 0));
            // cudaMemcpy3DParms p = {};
            // p.srcPtr = make_cudaPitchedPtr(dummy_d, /*pitch=*/1, /*xsize=*/1, /*ysize=*/1);
            // p.dstPtr = make_cudaPitchedPtr(dummy_h, /*pitch=*/1, /*xsize=*/1, /*ysize=*/1);
            // p.extent = make_cudaExtent(/*width bytes=*/1, 1, 1);
            // p.kind   = cudaMemcpyDeviceToHost;
            // cuda_err_chk(cudaGraphAddMemcpyNode(&this->memcpyNode[i], this->graph[i], nullptr, 0, &p));
            // cudaHostNodeParams hp = {};
            // hp.fn = host_cb;
            // hp.userData = nullptr;
            // const cudaGraphNode_t deps[] = { this->memcpyNode[i] };
            // cuda_err_chk(cudaGraphAddHostNode(&this->callbackNode[i], this->graph[i], deps, 1, &hp));
            // cuda_err_chk(cudaGraphInstantiate(&this->graphExec[i], this->graph[i], nullptr, nullptr, 0));
        }


        cuda_err_chk(cudaMallocManaged(&this->cmd, sizeof(volatile dma_cmd_t) * depth));
        cuda_err_chk(cudaMemset((void *) this->cmd, 0, sizeof(volatile dma_cmd_t) * depth));

        // avoid consistent page fault when polling
        int dev = 0;
        cudaGetDevice(&dev);
        void* p = (void*)(this->cmd);
        cudaMemLocation hostLoc{};
        hostLoc.type = cudaMemLocationTypeHost;
        hostLoc.id   = 0; // ignored for Host
        cuda_err_chk(cudaMemAdvise(p, sizeof(volatile dma_cmd_t) * depth, cudaMemAdviseSetPreferredLocation, hostLoc));
        cudaMemLocation devLoc{};
        devLoc.type = cudaMemLocationTypeDevice;
        devLoc.id   = dev;
        cuda_err_chk(cudaMemAdvise(p, sizeof(volatile dma_cmd_t) * depth, cudaMemAdviseSetAccessedBy, devLoc));
    }

    __host__
    ~AgileDmaSQHost() {
        cuda_err_chk(cudaFree(this->d_cmd_locks));
        free(this->cb_param);
        for(unsigned int i = 0; i < depth; ++i){
            cuda_err_chk(cudaStreamDestroy(this->stream[i]));
            cuda_err_chk(cudaEventDestroy(this->event[i]));
            // cuda_err_chk(cudaGraphDestroy(this->graph[i]));
            // cuda_err_chk(cudaGraphExecDestroy(this->graphExec[i]));
        }
        free(this->stream);
        free(this->event);
        // free(this->graph);
        // free(this->graphExec);
        // free(this->memcpyNode);
        // free(this->callbackNode);
        // free(this->cb_args);
        // cuda_err_chk(cudaDeviceSynchronize());
        cuda_err_chk(cudaFree((void *) this->cmd));
        // cuda_err_chk(cudaFree(dummy_d));
        // cuda_err_chk(cudaFreeHost(dummy_h));
    }
    __host__
    AgileDmaSQDevice * getDevicePtr(){
        if(this->ptr == nullptr){
            AgileDmaSQDevice * h_ptr = new AgileDmaSQDevice(cmd, d_cmd_locks, depth);
            cuda_err_chk(cudaMalloc(&this->ptr, sizeof(AgileDmaSQDevice)));
            cuda_err_chk(cudaMemcpy(this->ptr, h_ptr, sizeof(AgileDmaSQDevice), cudaMemcpyHostToDevice));
            delete h_ptr;
        }
        return this->ptr;
    }

};

class AgileDmaQueuePairDevice {
public:
    AgileDmaSQDevice *sq;
    AgileDmaCQDevice *cq;
};

class AgileDmaQueuePairHost {
public:
    AgileDmaSQHost *sq;
    AgileDmaCQHost *cq;

    AgileDmaQueuePairDevice * ptr = nullptr;

    __host__
    AgileDmaQueuePairDevice * getDevicePtr(){
        if(this->ptr == nullptr){
            auto h_ptr = new AgileDmaQueuePairDevice();
            h_ptr->sq = this->sq->getDevicePtr();
            h_ptr->cq = this->cq->getDevicePtr();
            cuda_err_chk(cudaMalloc(&this->ptr, sizeof(AgileDmaQueuePairDevice)));
            cuda_err_chk(cudaMemcpy(this->ptr, h_ptr, sizeof(AgileDmaQueuePairDevice), cudaMemcpyHostToDevice));
            delete h_ptr;
        }
        return this->ptr;
    }
};


class AgileDmaEngine : public DevicePtr<AgileDmaEngine>{
public:
    AgileDmaQueuePairDevice *queue_pair;
    unsigned int queue_num;
    
public:
    
    __host__
    AgileDmaEngine(AgileDmaQueuePairDevice *queue_pair, unsigned int queue_num) : queue_pair(queue_pair), queue_num(queue_num) {}
    
    __device__
    void submit(uint64_t src_addr, uint64_t dst_addr, unsigned int size, char flags, cmd_info_t &cmd_info, AgileLockChain *chain) {
        uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int q_idx = tid % queue_num;
        do {
            if(queue_pair[q_idx].sq->submitDMA_try(src_addr, dst_addr, size, flags, cmd_info, chain)){
                cmd_info.queue_id = q_idx;
                return;
            }
            q_idx = (q_idx + 1) % queue_num;
        } while (true);
    }

    __device__ 
    void wait(cmd_info_t &cmd_info){
        wait_cmd(&queue_pair[cmd_info.queue_id].sq->cmd_locks[cmd_info.cmd_id]);
    }
};
