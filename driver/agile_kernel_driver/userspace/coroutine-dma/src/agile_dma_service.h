#pragma once
#include <iostream>
#include <atomic>
#include <vector>
#include <stdexcept>
#include <system_error>
#include <pthread.h>
#include <sched.h>
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

typedef struct {
    uint64_t coroutine_idx;
    int cb_socket_sender;
    DmaRequest::promise_type *promise;
    AgileDmaCQHost *cq;
    uint32_t identifier;
} cuda_callback_param;

DmaRequest issue_and_consume_GPU(void * src, void * dst, size_t size, char flags, int cb_socket_sender,
    AgileDmaCQHost *cq, uint32_t identifier, cudaStream_t &stream){
    auto self = co_await DmaRequest::this_coro_t{};
    DmaRequest::promise_type promise = self.promise();
    uint64_t coroutine_idx = promise.coroutine_idx;
    // INFO("DMA transfer:", coroutine_idx);
    
    promise.t1 = std::chrono::high_resolution_clock::now();

    // Issue DMA command to GPU's DMA engine

    // printf("src: %p, dst: %p\n", src, dst);
    cuda_err_chk(cudaMemcpyAsync(dst, src, size, flags == DMA_CPU2GPU ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost, stream));

    promise.t2 = std::chrono::high_resolution_clock::now();

    // Set up CUDA callback parameters
    cuda_callback_param cb_param;
    cb_param.coroutine_idx = coroutine_idx;
    cb_param.cb_socket_sender = cb_socket_sender;
    cb_param.promise = &promise;
    cb_param.cq = cq;
    cb_param.identifier = identifier;
    cuda_err_chk(cudaLaunchHostFunc(stream, [](void *arg) {
        cuda_callback_param *cb_param = static_cast<cuda_callback_param *>(arg);
        // send(cb_param->cb_socket_sender, &cb_param->coroutine_idx, sizeof(cb_param->coroutine_idx), 0);
        cb_param->promise->t4 = std::chrono::high_resolution_clock::now();
        cb_param->cq->updateCpl(cb_param->identifier);
        cb_param->promise->t5 = std::chrono::high_resolution_clock::now();
    }, &cb_param));

    promise.t3 = std::chrono::high_resolution_clock::now();
    
    // Wait for the DMA to complete
    co_await std::suspend_always{};

    // promise.t5 = std::chrono::high_resolution_clock::now();
    // cq->updateCpl(identifier);
    // promise.t6 = std::chrono::high_resolution_clock::now();

    auto time0 = promise.t1 - promise.t0;
    auto time1 = promise.t2 - promise.t1;
    auto time2 = promise.t3 - promise.t2;
    auto time3 = promise.t4 - promise.t3;
    auto time4 = promise.t5 - promise.t4;

    // printf("%lld, %lld, %lld, %lld, %lld\n",
    //     time0.count(), time1.count(), time2.count(), time3.count(),
    //     time4.count());

    co_return;
}

DmaRequest issue_and_consume_CPU(int fd, struct dma_command cmd, AgileDmaCQHost *cq, uint32_t identifier){
    auto self = co_await DmaRequest::this_coro_t{};
    DmaRequest::promise_type promise = self.promise();
    cmd.coroutine_idx = promise.coroutine_idx;
    // auto krnl_start = std::chrono::high_resolution_clock::now();
    // Issue DMA command to CPU's DMA engine
    if(ioctl(fd, IOCTL_SUBMIT_DMA_CMD, &cmd) < 0){
    // if(ioctl(fd, IOCTL_TEST_DMA_EMU, &cmd) < 0){
        debug("Failed to issue DMA command");
    }
    // auto krnl_end = std::chrono::high_resolution_clock::now();

    // Wait for the DMA to complete
    co_await std::suspend_always{};
    // promise.dma_finish = std::chrono::high_resolution_clock::now();

    // Post processing after DMA
    cq->updateCpl(identifier);
    // auto finish_time = std::chrono::high_resolution_clock::now();

    // auto dma_issue_time = promise.dma_issue - promise.generation;
    // auto dma_finish_time = promise.dma_finish - promise.dma_issue;
    // auto post_processing_time = finish_time - promise.dma_finish;

    // printf("%lld, %lld, %lld, %lld\n", dma_issue_time.count(), dma_finish_time.count(), post_processing_time.count(), (krnl_end - krnl_start).count());

    co_return;
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
    AgileHostWorker(int worker_id, int k_event, int u_event, int c_event, int stop_sig, struct dma_cpl_queue * cpl_queue){
        this->worker_id = worker_id;
        this->k_event = k_event;
        this->u_event = u_event;
        this->c_event = c_event;
        this->stop_sig = stop_sig;
        this->cpl_queue = cpl_queue;
        this->req_idx = 0;

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

    int waitEvents(){
        event_count = epoll_wait(efd, events, EVENT_ARRAY_SIZE, -1);
        return event_count;
    }

    int processEvents(){
        for (int i = 0; i < event_count; ++i) {
            if(events[i].data.fd == u_event){ // receiving signal from the monitor threads
                printf("error\n");
                // DmaRequest req;
                // size_t r = recv(u_event, &req, sizeof(req), 0);
                // if (r == sizeof(req)) {
                //     debug("Received user event, id:", req_idx, "r:", r, "sizeof(DmaRequest):", sizeof(DmaRequest));
                //     req.handle.promise().coroutine_idx = req_idx;
                //     // req.handle.promise().dma_issue = std::chrono::high_resolution_clock::now();
                //     req.handle.resume();
                //     requests[req_idx++] = req;
                // } else {
                //     debug("Partial read or no data available on user event socket, r:", r);
                // }
            }else if(events[i].data.fd == k_event) { // receiving signal from kernel
                uint64_t val = 0;
                read(k_event, &val, sizeof(val));
                debug("Received kernel event");
                // processKernelEvent(val);
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
            debug("Process kernel event for coroutine_idx:", coroutine_idx);
            this->cpl_queue->data.entry[this->cpl_queue->head].coroutine_idx = 0;
            if(requests.find(coroutine_idx) == requests.end()){
                MTLOG_ERROR("Cannot find coroutine idx:", coroutine_idx);
                continue;
            }
            auto req = requests[coroutine_idx];
            req.handle.resume();
            this->cpl_queue->head = (this->cpl_queue->head + 1) % this->cpl_queue->depth;
            requests.erase(coroutine_idx);
        }
    }

    void insertRequest(DmaRequest req){
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

private:
    int worker_id;
    int efd;
    // int fd;
    int k_event; // eventfd, kernel event for notifying completion of DMA command
    int u_event; // socket, user event for resume target coroutine to submit the DMA command
    int c_event; // eventfd, cuda event for notifying completion of DMA command
    int stop_sig;
    struct dma_cpl_queue * cpl_queue; // completion queue for this worker thread
    
    std::map<uint64_t, DmaRequest> requests;
    mutable std::shared_mutex mx;

    uint64_t req_idx;
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

private:
    AgileHostWorker **workers;
    struct socket_pair *u_socket_pairs;
    struct socket_pair *c_socket_pairs;
    unsigned int num_workers;
    std::atomic_uint g_pair_idx{0};
};


class AgileHostMonitor {
public:

    AgileHostMonitor(int monitor_id, int fd, uint64_t dram_start_offset, uint64_t hbm_start_offset, AgileDmaQueuePairHost *queue_pairs, unsigned int sq_num, AgileDispatcher *dispatcher): monitor_id(monitor_id), fd(fd), dram_start_offset(dram_start_offset), hbm_start_offset(hbm_start_offset), queue_pairs(queue_pairs), sq_num(sq_num), dispatcher(dispatcher) {}

    ~AgileHostMonitor(){
    }

    void start(){
        t_ = std::jthread([this](std::stop_token st){ monitoring(st); });
    }

    void stop(){
        t_.request_stop();
    }

    void setHostPtr(void *ptr) { host_ptr = ptr; }
    void setDevicePtr(void *ptr) { device_ptr = ptr; }

private:
    void monitoring(std::stop_token st){
        // pin_current_thread_to(this->monitor_id*2 + 4);
        int cpu_id = sched_getcpu();
        uint32_t cmd_idx = 0;
        uint64_t waste_chk = 0;
        INFO("start monitoring on CPU", cpu_id);
        auto t0 = std::chrono::high_resolution_clock::now();
        while (!st.stop_requested()) {
            for(unsigned int i = 0; i < sq_num; ++i){
                unsigned int cmd_pos = queue_pairs[i].sq->tail % queue_pairs[i].sq->depth;
                volatile uint32_t *trigger = &queue_pairs[i].sq->cmd[cmd_pos].trigger;
                if(*trigger == 1){
                    auto start_time = std::chrono::high_resolution_clock::now();
                    int worker_idx, u_sender, c_sender;
                    AgileHostWorker *worker;
                    dispatcher->getSenderSocket(&worker, &worker_idx, &u_sender, &c_sender);
                    // printf("worker: %p\n", worker);
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
                    // printf("src_addr: %llu, dst_addr: %llu\n", queue_pairs[i].sq->data[cmd_pos].src_addr, queue_pairs[i].sq->data[cmd_pos].dst_addr);

                    
                    uint32_t identifier = queue_pairs[i].sq->cmd[cmd_pos].identifier;
                    // printf("got cmd lock: %p\n", queue_pairs[i].sq->data[cmd_pos].lock);
                    auto req = issue_and_consume_GPU(src, dst, queue_pairs[i].sq->cmd[cmd_pos].size, queue_pairs[i].sq->cmd[cmd_pos].flags, c_sender, queue_pairs[i].cq, identifier, queue_pairs[i].sq->stream[cmd_pos]);
                    // worker->insertRequest(req);
                    req.handle.promise().t0 = std::chrono::high_resolution_clock::now();
                    req.handle.resume();
                    *trigger = 0;
                    // req.handle.resume();
                    // req.handle.promise().coroutine_idx = identifier; // use identifier as coroutine_idx for CPU coroutine
                    // req.handle.resume();
                    // struct dma_command cmd;
                    // cmd.src_addr = src_addr;
                    // cmd.dst_addr = dst_addr;
                    // cmd.size = queue_pairs[i].sq->data[cmd_pos].size;
                    // cmd.cpl_queue_idx = worker_idx; // TODO: support multiple completion queues
                    // cmd.dma_channel_idx = worker_idx; // TODO: support multiple DMA channels
                    // auto req = issue_and_consume_CPU(this->fd, cmd, queue_pairs[i].cq, identifier);
                    // req.handle.promise().generation = std::chrono::high_resolution_clock::now();
                    // send the handle to the working thread
                    // send(u_sender, &req, sizeof(req), 0);
                    queue_pairs[i].sq->tail = (queue_pairs[i].sq->tail + 1) % queue_pairs[i].sq->depth;
                    auto end_time = std::chrono::high_resolution_clock::now();
                    auto duration = end_time - start_time;
                    // printf("%p, %d, %lld, %lld, %lld\n", this, cmd_idx++, (start_time - t0).count(), duration.count(), waste_chk);
                    // printf("monitor_idx: %d pos: %d, src_addr: %llx, dst_addr: %llx, size: %d, trigger: %d\n", this->monitor_id, cmd_pos, queue_pairs[i].sq->data[cmd_pos].src_addr,  queue_pairs[i].sq->data[cmd_pos].dst_addr, queue_pairs[i].sq->data[cmd_pos].size, queue_pairs[i].sq->data[cmd_pos].trigger);
                    waste_chk = 0;
                }else{
                    usleep(10);
                    waste_chk++;
                    if(waste_chk >= 10000){
                        printf("monitor_idx: %d pos: %d, src_addr: %llx, dst_addr: %llx, size: %d, trigger: %d\n", this->monitor_id, cmd_pos, queue_pairs[i].sq->cmd[cmd_pos].src_addr,  queue_pairs[i].sq->cmd[cmd_pos].dst_addr, queue_pairs[i].sq->cmd[cmd_pos].size, *trigger);
                        waste_chk = 0;
                    }
                    // INFO("running...");
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
};
