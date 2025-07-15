#ifndef AGILE_LOGGER
#define AGILE_LOGGER

class alignas(64) AgileLogger{
public:
    unsigned int issued_read;
    unsigned int finished_read;
    unsigned int issued_write;
    unsigned int finished_write;

    unsigned int write_threads;

    unsigned int gpu_cache_hit;
    unsigned int gpu_cache_miss;
    unsigned int gpu_cache_evict;
    unsigned int prefetch_hit;
    unsigned int prefetch_relaxed_hit;
    unsigned int prefetch_relaxed_miss;
    unsigned int prefetch_issue;
    unsigned int runtime_issue;
    unsigned int warp_master_wait;

    unsigned int cpu_cache_hit;
    unsigned int cpu_cache_miss;
    unsigned int cpu_cache_evict;

    unsigned int gpu2cpu;
    unsigned int cpu2buf;
    unsigned int cpu2gpu;
    unsigned int gpu2nvme;
    unsigned int cpu2nvme;

    unsigned int waiting;
    unsigned int service;
    unsigned int finished_block;
    unsigned int finished_agile_warp;

    unsigned int self_propagate;

    unsigned int propogate_time;
    unsigned int propogate_count;
    unsigned int appendbuf_count;
    
    unsigned int waitTooMany;
    unsigned int buffer_localhit;
    unsigned int wating_buffer;
    unsigned int finish_buffer;
    unsigned int deadlock_check;

    unsigned int push_in_table;
    unsigned int pop_in_table;

    unsigned int find_new_cacheline;

    unsigned int attempt_fail;

//     unsigned int cmd_count[128];
//     unsigned int cpl_count[128];
//     unsigned int last_cq_pos[128];
//     unsigned int last_sq_pos[128];
//     unsigned int curr_sq_pos[128];
//     unsigned int curr_cq_pos[128];
//     unsigned int cq_waiting[128];
//     unsigned int cq_running[128];
//     unsigned int gap[128];
};

#endif