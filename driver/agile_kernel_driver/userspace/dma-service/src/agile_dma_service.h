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
#include "agile_dma_queue.hpp"
#include "agile_helper.h"

#include "io_utils.h"

inline void pin_current_thread_to(int cpu) {
    cpu_set_t mask; CPU_ZERO(&mask);
    long nconf = sysconf(_SC_NPROCESSORS_CONF);

    CPU_SET(cpu, &mask);
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
    if (rc != 0) throw std::system_error(rc, std::generic_category(), "pthread_setaffinity_np");
}


class AgileHostWorker{
public:
    AgileHostWorker(int worker_id, AgileGpuDmaSQHost **sqs, AgileGpuDmaCQHost **cqs, unsigned int queue_num) : worker_id(worker_id), stop_flag(0), sqs(sqs), cqs(cqs), queue_num(queue_num){ 
    }

    ~AgileHostWorker(){
    }

    int start(){
        uint32_t identifier;
        while(!this->stop_flag){
            for(uint32_t i = 0; i < queue_num; ++i){
                if(sqs[i]->pollCudaEvents(&identifier)){
                    // DEBUG("Worker", worker_id, "detected completion on SQ", sqs[i], "identifier", identifier);
                    cqs[i]->updateCpl(identifier);
                }
            }
        }
        return 0;
    }

    uint32_t ID(){
        return worker_id;
    }

    void stop(){
        this->stop_flag = 1;
    }

private:
    int worker_id;
    int stop_flag;

    AgileGpuDmaSQHost **sqs;
    AgileGpuDmaCQHost **cqs;
    unsigned int queue_num;

    // std::map<uint64_t, DmaRequest> requests;
    // mutable std::shared_mutex mx;
    // uint64_t req_idx = 1;
};


class AgileHostMonitor {
public:

    AgileHostMonitor(int monitor_id, int fd, uint64_t dram_start_offset, uint64_t hbm_start_offset, AgileDmaSQHost ** sq_array, unsigned int sq_num): monitor_id(monitor_id), fd(fd), dram_start_offset(dram_start_offset), hbm_start_offset(hbm_start_offset), sq_array(sq_array), sq_num(sq_num) {
    }

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
        int node = numa_node_of_cpu(cpu_id);
        INFO("start monitoring on CPU", cpu_id, "NUMA node", node);
        while (!st.stop_requested()) {
            for(unsigned int i = 0; i < sq_num; ++i){
                unsigned int cmd_count = sq_array[i]->poll(fd);
            }
        }
    }

    int monitor_id;
    int fd;
    uint64_t dram_start_offset;
    uint64_t hbm_start_offset;
    std::jthread t_;
    AgileDmaSQHost ** sq_array;
    unsigned int sq_num;
    void * host_ptr;
    void * device_ptr;
};
