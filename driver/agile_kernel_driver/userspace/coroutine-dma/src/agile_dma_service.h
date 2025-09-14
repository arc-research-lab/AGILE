#pragma once
#include <iostream>
#include <atomic>
#include <vector>
#include <stdexcept>
#include <system_error>
#include <pthread.h>
#include <sched.h>
#include <numa.h>
#include <sys/epoll.h> 
#include <sys/eventfd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <unistd.h>
#include <shared_mutex>



#include "agile_coroutine.h"
#include "agile_dma.hpp"
#include "agile_helper.h"

#include "io_utils.h"

static inline uint64_t ts_to_ns(const struct timespec* ts) {
    return (uint64_t)ts->tv_sec * 1000000000ull + (uint64_t)ts->tv_nsec;
}


struct dma_cpl_queue {
    uint32_t idx;
    union dma_cpl_queue_data data;
    uint32_t head;
    uint32_t tail;
    uint32_t depth;
};

struct dma_cpl_queues_info {
    struct dma_cpl_queue *queues;
    int num_queues;
};

#define SOCKET_SEND 1
#define SOCKET_RECEIVE 0

struct socket_pair {
    int msg_sock[2]; // [1]: send; [0]: receive
};

DmaRequest issue_and_consume_GPU(void * src, void * dst, size_t size, 
  char flags, AgileDmaCQHost *cq, uint32_t identifier, cudaStream_t &stream, 
  cudaEvent_t &event, cuda_callback_param *cb_param) {
    auto promise = co_await DmaRequest::this_coro_t{};
    // DmaRequest::promise_type promise = self.promise();
    uint64_t coroutine_idx = promise->coroutine_idx;
    cuda_err_chk(cudaMemcpyAsync(dst, src, size, \
        flags == DMA_CPU2GPU ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost, stream));
    cuda_err_chk(cudaEventRecord(event, stream));
    co_return;
}

// DmaRequest issue_and_consume_GPU_bak(void * src, void * dst, size_t size, char flags, int cb_socket_sender, \
// AgileDmaCQHost *cq, uint32_t identifier, \
// cudaStream_t &stream, cudaGraphExec_t &graphExec, cudaGraphNode_t &memcpyNode, cudaGraphNode_t &callbackNode, DmaCallbackArgs &cb_args){
//     auto promise = co_await DmaRequest::this_coro_t{};
//     uint64_t coroutine_idx = promise->coroutine_idx;

//     // Issue DMA command to GPU's DMA engine
//     cudaMemcpy3DParms p{};
//     p.srcPtr = make_cudaPitchedPtr(src, size, size, 1);
//     p.dstPtr = make_cudaPitchedPtr(dst, size, size, 1);
//     p.extent = make_cudaExtent(size, 1, 1);
//     p.kind   = ((flags == DMA_CPU2GPU) ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost);

//     cuda_err_chk(cudaGraphExecMemcpyNodeSetParams(graphExec, memcpyNode, &p)); // GPUassert: invalid argument

//     cb_args.coroutine_idx = coroutine_idx;
//     cb_args.promise = promise;
//     cb_args.cq = cq;
//     cb_args.identifier = identifier;

//     cudaHostNodeParams hp{};
//     hp.fn = host_cb;
//     hp.userData = &cb_args;  
//     cuda_err_chk(cudaGraphExecHostNodeSetParams(graphExec, callbackNode, &hp));

//     cuda_err_chk(cudaGraphLaunch(graphExec, stream));

//     co_await std::suspend_always{};

//     co_return;
// }

DmaRequest issue_and_consume_CPU(int fd, struct dma_command cmd, AgileDmaCQHost *cq, uint32_t identifier){
    auto promise = co_await DmaRequest::this_coro_t{};
    printf("promise in issue_and_consume_CPU: id: %d\n", promise->coroutine_idx);
    cmd.coroutine_idx = promise->coroutine_idx;

    clock_gettime(CLOCK_BOOTTIME, &promise->init_time);

    if(ioctl(fd, IOCTL_SUBMIT_DMA_CMD, &cmd) < 0){
        debug("Failed to issue DMA command");
    }

    clock_gettime(CLOCK_BOOTTIME, &promise->finish_issue_time);

    // Wait for the DMA to complete
    co_await std::suspend_always{};

    clock_gettime(CLOCK_BOOTTIME, &promise->after_dma_awake_time);

    cq->updateCpl(identifier);

    clock_gettime(CLOCK_BOOTTIME, &promise->end_time);

    uint64_t time0 = cmd.t0 - ts_to_ns(&promise->init_time);
    uint64_t time1 = cmd.t1 - cmd.t0; // in kernel
    uint64_t time2 = cmd.t2 - cmd.t1; // dma mapping
    uint64_t time3 = cmd.t3 - cmd.t2; // dma transfer
    uint64_t time4 = cmd.t4 - cmd.t3; // dma unmapping
    uint64_t time5 = cmd.t5 - cmd.t4;
    uint64_t time6 = cmd.t6 - cmd.t5;
    uint64_t time7 = cmd.t7 - cmd.t6; 
    uint64_t time8 = cmd.t8 - cmd.t7; 
    uint64_t time9 = ts_to_ns(&promise->finish_issue_time) - cmd.t8;
    uint64_t time10 = promise->dma_callback_time - ts_to_ns(&promise->finish_issue_time);
    uint64_t time11 = ts_to_ns(&promise->after_dma_awake_time) - promise->dma_callback_time;
    uint64_t time12 = ts_to_ns(&promise->end_time) - ts_to_ns(&promise->after_dma_awake_time);

    printf("%lld, %lld, %lld, %lld, %lld, %lld, %lld, %lld, %lld, %lld, %lld, %lld, %lld\n",
           time0, time1, time2, time3, time4, time5, time6, time7, time8, time9, time10, time11, time12);
}


inline void pin_current_thread_to(int cpu) {
    cpu_set_t mask; CPU_ZERO(&mask);
    long nconf = sysconf(_SC_NPROCESSORS_CONF);

    CPU_SET(cpu, &mask);
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
    if (rc != 0) throw std::system_error(rc, std::generic_category(), "pthread_setaffinity_np");
}


#define EVENT_ARRAY_SIZE 8 // TODO: change this
class AgileHostWorker{
public:
    AgileHostWorker(int worker_id, int k_event, int u_event, int c_event, int stop_sig, struct dma_cpl_queue * cpl_queue, AgileDmaQueuePairHost *queue_pairs, unsigned int queue_num){
        this->worker_id = worker_id;
        this->k_event = k_event;
        this->u_event = u_event;
        this->c_event = c_event;
        this->stop_sig = stop_sig;
        this->stop_flag = 0;
        this->cpl_queue = cpl_queue;
        this->req_idx = 1;
        this->queue_pairs = queue_pairs;
        this->queue_num = queue_num;

        this->efd = epoll_create1(EPOLL_CLOEXEC);
        struct epoll_event ev = {0};
        ev.events = EPOLLIN;
        ev.data.fd = k_event;
        epoll_ctl(this->efd, EPOLL_CTL_ADD, k_event, &ev);

        ev.events = EPOLLIN;
        ev.data.fd = u_event;
        epoll_ctl(this->efd, EPOLL_CTL_ADD, u_event, &ev);

        ev.events = EPOLLIN;
        ev.data.fd = c_event;
        epoll_ctl(this->efd, EPOLL_CTL_ADD, c_event, &ev);

        ev.events = EPOLLIN;
        ev.data.fd = stop_sig;
        epoll_ctl(this->efd, EPOLL_CTL_ADD, stop_sig, &ev);
    }

    ~AgileHostWorker(){
        ::close(this->efd);
    }

    int waitEvents_bak(){

        while(!this->stop_flag){
            for(uint32_t i = 0; i < queue_num; ++i){
                unsigned int cmd_pos = queue_pairs[i].sq->head;
                volatile uint32_t *trigger = &queue_pairs[i].sq->cmd[cmd_pos].trigger;
                if(*trigger == 2){
                    // Process the command
                    if(cudaEventQuery(queue_pairs[i].sq->event[cmd_pos]) == cudaSuccess){
                        // Mark the command as finished
                        uint32_t identifier = queue_pairs[i].sq->cmd[cmd_pos].identifier;
                        *trigger = 0;
                        queue_pairs[i].cq->updateCpl(identifier);
                        queue_pairs[i].sq->head = (queue_pairs[i].sq->head + 1) % queue_pairs[i].sq->depth;
                    }
                }
            }
        }
        
        return 0;
    }

    int waitEvents(){
        event_count = epoll_wait(efd, events, EVENT_ARRAY_SIZE, -1);
        return event_count;
    } 

    int processEvents(){
        for (int i = 0; i < event_count; ++i) {
            if(events[i].data.fd == u_event){ // receiving signal from the monitor threads
                printf("error\n");
            }else if(events[i].data.fd == k_event) { // receiving signal from kernel
                uint64_t val = 0;
                read(k_event, &val, sizeof(val));
                debug("Received kernel event");
                processKernelEvent(val);
            }else if(events[i].data.fd == c_event){ // receiving signal from CUDA-managed thread
                uint64_t coroutine_idx = 0;
                read(c_event, &coroutine_idx, sizeof(coroutine_idx));
                debug("Received cuda event: ", coroutine_idx, " finished");
                if(requests.find(coroutine_idx) == requests.end()){
                    printf("Cannot find coroutine idx: %d\n", coroutine_idx);
                    continue;
                }
                auto req = requests[coroutine_idx];
                req.handle.resume();
                // requests.erase(coroutine_idx);
                this->removeRequest(coroutine_idx);
            }else if(events[i].data.fd == stop_sig) { // receiving stop signal
                return -1;
            }else{
                debug("Unknown event fd");
            }
        }
        return 0;
    }

    void processKernelEvent(uint64_t val){
        for(unsigned int i = 0; i < val; ++i){
            uint64_t coroutine_idx = this->cpl_queue->data.entry[this->cpl_queue->head].coroutine_idx;
            uint64_t finish_time = this->cpl_queue->data.entry[this->cpl_queue->head].finish_time;
            INFO("Process kernel event for coroutine_idx:", coroutine_idx, " at time ", finish_time);
            this->cpl_queue->data.entry[this->cpl_queue->head].coroutine_idx = 0;
            this->cpl_queue->data.entry[this->cpl_queue->head].finish_time = 0;
            if(requests.find(coroutine_idx) == requests.end()){
                MTLOG_ERROR("Cannot find coroutine idx:", coroutine_idx);
                continue;
            }
            auto req = requests[coroutine_idx];
            req.handle.promise().dma_callback_time = finish_time;
            auto p = req.handle.promise();
            req.handle.resume();
            this->cpl_queue->head = (this->cpl_queue->head + 1) % this->cpl_queue->depth;
            this->removeRequest(coroutine_idx);
        }
    }

    void insertRequest(DmaRequest &req){
        std::unique_lock lock(mx);
        req.handle.promise().coroutine_idx = req_idx;
        requests[req_idx++] = req;
    }

    void removeRequest(uint64_t coroutine_idx){
        std::unique_lock lock(mx);
        requests.erase(coroutine_idx);
    }

    uint32_t ID(){
        return worker_id;
    }

    void stop(){
        this->stop_flag = 1;
    }

private:
    int worker_id;
    int efd;
    // int fd;
    int k_event; // eventfd, kernel event for notifying completion of DMA command
    int u_event; // socket, user event for resume target coroutine to submit the DMA command
    int c_event; // eventfd, cuda event for notifying completion of DMA command
    int stop_sig;
    int stop_flag;
    struct dma_cpl_queue * cpl_queue; // completion queue for this worker thread

    AgileDmaQueuePairHost *queue_pairs;
    unsigned int queue_num;

    std::map<uint64_t, DmaRequest> requests;
    mutable std::shared_mutex mx;

    uint64_t req_idx = 1;
    struct epoll_event events[EVENT_ARRAY_SIZE] = {0};
    int event_count;
};


class AgileDispatcher {
public:
    AgileDispatcher(AgileHostWorker **workers, struct socket_pair *u_socket_pairs, struct socket_pair *c_socket_pairs, uint32_t num_workers) {
        this->workers = workers;
        this->u_socket_pairs = u_socket_pairs;
        this->c_socket_pairs = c_socket_pairs;
        this->num_workers = num_workers;
        this->g_pair_idx = 0;
    }

    ~AgileDispatcher() {
    }

    void getSenderSocket(AgileHostWorker **worker, int* worker_id, int * u_sender, int * c_sender){
        unsigned int pair_idx = g_pair_idx.fetch_add(1) % num_workers;
        *u_sender = u_socket_pairs[pair_idx].msg_sock[SOCKET_SEND];
        *c_sender = c_socket_pairs[pair_idx].msg_sock[SOCKET_SEND];
        *worker_id = pair_idx;
        *worker = workers[pair_idx];
    }

    AgileHostWorker **workers;
    struct socket_pair *u_socket_pairs;
    struct socket_pair *c_socket_pairs;
    unsigned int num_workers;
    std::atomic_uint g_pair_idx{0};
};

class AgileHostMonitor {
public:

    AgileHostMonitor(int monitor_id, int fd, uint64_t dram_start_offset, uint64_t hbm_start_offset, AgileDmaQueuePairHost *queue_pairs, unsigned int sq_num, AgileDispatcher *dispatcher): monitor_id(monitor_id), fd(fd), dram_start_offset(dram_start_offset), hbm_start_offset(hbm_start_offset), queue_pairs(queue_pairs), sq_num(sq_num), dispatcher(dispatcher) {
        // this->logs = (monitor_log_item *) malloc(sizeof(monitor_log_item) * 16384);
    }

    ~AgileHostMonitor(){
        // free(this->logs);
    }

    void start(){
        t_ = std::jthread([this](std::stop_token st){ monitoring(st); });
    }

    void stop(){
        t_.request_stop();
    }

    void setHostPtr(void *ptr) { host_ptr = ptr; }
    void setDevicePtr(void *ptr) { device_ptr = ptr; }

    // void printLogs(const char * filename){
    //     FILE * fp = fopen(filename, "w");
    //     if(fp == nullptr){
    //         return;
    //     }
    //     fprintf(fp, "cmd_idx,t0,t1,t2,t3,t4,t5,t6\n");
    //     for(unsigned int i = 0; i < log_count; ++i){
    //         auto log_item = &this->logs[i];
    //         if(log_item->t0.time_since_epoch().count() == 0) break;
    //         auto time0 = log_item->t1 - log_item->t0;
    //         auto time1 = log_item->t2 - log_item->t1;
    //         auto time2 = log_item->t3 - log_item->t2;
    //         auto time3 = log_item->t4 - log_item->t3;
    //         auto time4 = log_item->t5 - log_item->t4;
    //         auto time5 = log_item->t6 - log_item->t5;
    //         fprintf(fp, "%d,%lld,%lld,%lld,%lld,%lld,%lld\n", i,
    //             time0.count(), time1.count(), time2.count(), time3.count(),
    //             time4.count(), time5.count());
    //     }
    //     fclose(fp);
    // }

private:
    void monitoring(std::stop_token st){
        // pin_current_thread_to(this->monitor_id*2 + 4);
        int cpu_id = sched_getcpu();
        int node = numa_node_of_cpu(cpu_id);
        INFO("start monitoring on CPU", cpu_id, "NUMA node", node);
        auto t0 = std::chrono::high_resolution_clock::now();
        while (!st.stop_requested()) {
            for(unsigned int i = 0; i < sq_num; ++i){
                unsigned int cmd_pos = queue_pairs[i].sq->tail;
                volatile uint32_t *trigger = &queue_pairs[i].sq->cmd[cmd_pos].trigger;
                if(*trigger == 1){
                    void * src;
                    void * dst;
                    uint64_t src_addr = 0;
                    uint64_t dst_addr = 0;
                    if(queue_pairs[i].sq->cmd[cmd_pos].flags == DMA_CPU2GPU){
                        debug("Detected DMA command at SQ", i, "pos", cmd_pos, ": CPU to GPU, socket:", u_sender);
                        src = host_ptr + queue_pairs[i].sq->cmd[cmd_pos].src_addr;
                        src_addr = dram_start_offset + queue_pairs[i].sq->cmd[cmd_pos].src_addr;
                        dst = device_ptr + queue_pairs[i].sq->cmd[cmd_pos].dst_addr;
                        dst_addr = hbm_start_offset + queue_pairs[i].sq->cmd[cmd_pos].dst_addr;
                    }else{
                        debug("Detected DMA command at SQ", i, "pos", cmd_pos, ": GPU to CPU, socket:", u_sender);
                        src = device_ptr + queue_pairs[i].sq->cmd[cmd_pos].src_addr;
                        src_addr = hbm_start_offset + queue_pairs[i].sq->cmd[cmd_pos].src_addr;
                        dst = host_ptr + queue_pairs[i].sq->cmd[cmd_pos].dst_addr;
                        dst_addr = dram_start_offset + queue_pairs[i].sq->cmd[cmd_pos].dst_addr;
                    }
                    uint32_t identifier = queue_pairs[i].sq->cmd[cmd_pos].identifier;
/* GPU DMA engine

                    auto temp_sq = queue_pairs[i].sq;
                    auto req = issue_and_consume_GPU(src, dst, temp_sq->cmd[cmd_pos].size, temp_sq->cmd[cmd_pos].flags, queue_pairs[i].cq, identifier, temp_sq->stream[cmd_pos], temp_sq->event[cmd_pos], &temp_sq->cb_param[cmd_pos]);
                    *trigger = 2;
                    req.handle.resume();
*/
                    
                    struct dma_command cmd;
                    cmd.src_addr = src_addr;
                    cmd.dst_addr = dst_addr;
                    cmd.size = queue_pairs[i].sq->cmd[cmd_pos].size;
                    cmd.cpl_queue_idx = 0;
                    cmd.dma_channel_idx = 0;
                    auto req = issue_and_consume_CPU(this->fd, cmd, queue_pairs[i].cq, identifier);
                    *trigger = 0;
                    dispatcher->workers[0]->insertRequest(req);
                    // auto promise = req.getPromise();
                    // promise->dma_callback_time = 10086;
                    // printf("promise in monitoring: %p\n", promise);
                    req.handle.resume();
                    queue_pairs[i].sq->tail = (queue_pairs[i].sq->tail + 1) % queue_pairs[i].sq->depth;
                }
            }
        }
    }

    int monitor_id;
    int fd;
    uint64_t dram_start_offset;
    uint64_t hbm_start_offset;
    std::jthread t_;
    AgileDispatcher *dispatcher;
    AgileDmaQueuePairHost * queue_pairs;
    void * host_ptr;
    void * device_ptr;
    unsigned int sq_num;

    // monitor_log_item * logs;
    uint32_t log_count = 0;

};
