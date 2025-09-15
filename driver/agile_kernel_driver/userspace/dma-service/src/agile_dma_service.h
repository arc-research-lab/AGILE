#pragma once
#include <iostream>
#include <atomic>
#include <vector>
#include <stdexcept>
#include <system_error>
#include <pthread.h>
#include <sched.h>
#include <numa.h>
#include <sys/types.h>
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


struct monitor_log_t {
    uint64_t start_time;
    uint32_t cmd_count;
    uint64_t duration;
};

class AgileHostMonitor {
public:

    AgileHostMonitor(int monitor_id, int fd, uint64_t dram_start_offset, uint64_t hbm_start_offset, AgileDmaSQHost ** sq_array, unsigned int sq_num)
    : monitor_id(monitor_id), fd(fd), dram_start_offset(dram_start_offset), hbm_start_offset(hbm_start_offset), sq_array(sq_array), sq_num(sq_num), host_ptr(nullptr), device_ptr(nullptr) {
        logs = (struct monitor_log_t *) malloc(sizeof(struct monitor_log_t) * 32768);
        memset(logs, 0, sizeof(struct monitor_log_t) * 32768);
    }

    ~AgileHostMonitor(){
        free(logs);
    }

    void start(){
        t_ = std::jthread([this](std::stop_token st){ monitoring(st); });
    }

    void stop(){
        t_.request_stop();
    }

    void setHostPtr(void *ptr) { host_ptr = ptr; }
    void setDevicePtr(void *ptr) { device_ptr = ptr; }

    void printLogs(){
        uint64_t total_time = 0;
        uint32_t total_cmds = 0;
        uint64_t prev_start_time = logs[0].start_time;
        uint64_t interval = 0;
        for(uint32_t i = 0; i < log_count; ++i){
            total_time += logs[i].duration;
            total_cmds += logs[i].cmd_count;
            // std::cout << "Monitor " << monitor_id << " Log " << i << ": Start Interval " << logs[i].start_time - prev_start_time << " ns, Cmd Count " << logs[i].cmd_count << ", Duration " << logs[i].duration << " ns" << std::endl;
            interval += logs[i].start_time - prev_start_time;
            prev_start_time = logs[i].start_time;
        }

        double predict_bw = (4096.0 / ((((double)interval)) / ((double)(log_count - 1)))) * (((double)total_cmds) / ((double)log_count));
        // average time per command
        if(total_cmds > 0){
            std::cout << "Monitor " << monitor_id << ": Total Commands " << total_cmds << ", Total IOCTL: " << log_count << ", Total Time " << total_time << " ns, Average Time per Command " << ((double)total_time / (double)total_cmds) << " ns, " << "Interval " << interval / (log_count - 1) << " ns, " << "Predicted Bandwidth " << predict_bw << " GB/s" << std::endl;
        }
    }

private:
    void monitoring(std::stop_token st){
        // pin_current_thread_to(this->monitor_id*2 + 4);
        int cpu_id = sched_getcpu();
        int node = numa_node_of_cpu(cpu_id);
        INFO("start monitoring on CPU", cpu_id, "NUMA node", node);
        while (!st.stop_requested()) {
            for(unsigned int i = 0; i < sq_num; ++i){
                auto start = std::chrono::high_resolution_clock::now();
                unsigned int cmd_count = sq_array[i]->poll(fd, this->monitor_id);
                auto end = std::chrono::high_resolution_clock::now();
                if(cmd_count > 0 && log_count < 32768){
                    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
                    logs[log_count].start_time = std::chrono::duration_cast<std::chrono::nanoseconds>(start.time_since_epoch()).count();
                    logs[log_count].cmd_count = cmd_count;
                    logs[log_count].duration = duration;
                    log_count++;
                }
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
    

    struct monitor_log_t *logs;
    uint32_t log_count = 0;


};
