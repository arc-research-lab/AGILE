#include <cstdio>
#include <string>
#include <chrono>
#include <sstream>

#include <CLI/CLI.hpp>

#include "agile_host.h"
#include "agile_buf_shared.h"
#include "cache_impl.h"
#include "table_impl.h"
#include "demo_ipc.h"
#include <logger.hpp>

#define CPU_CACHE_IMPL DisableCPUCache
#define WRITE_TABLE_IMPL DisableShareTable
#define GPU_CACHE_IMPL SimpleGPUCache<CPU_CACHE_IMPL, WRITE_TABLE_IMPL>

#define AGILE_CTRL AgileCtrl<GPU_CACHE_IMPL, CPU_CACHE_IMPL, WRITE_TABLE_IMPL>
#define AGILE_HOST AgileHost<GPU_CACHE_IMPL, CPU_CACHE_IMPL, WRITE_TABLE_IMPL>
#define AGILE_BUF_ARR AgileBufArrayShared<GPU_CACHE_IMPL, CPU_CACHE_IMPL, WRITE_TABLE_IMPL>


__device__ void compute_krnl(bool enable, AGILE_BUF_ARR &bufArr, unsigned int compute_sim, unsigned int compute_itr){
    if(enable){
        __syncthreads();
        for(unsigned int i = 0; i < compute_itr; ++i){
            if(threadIdx.x < bufArr.ctrl->buf_size / sizeof(unsigned int)){
                ((unsigned int *)((bufArr.buf[0]).data))[threadIdx.x] = ((unsigned int *)((bufArr.buf[0]).data))[threadIdx.x] * compute_sim;
            }
        }
        __syncthreads();
    }
}

__device__ void issue_load(bool enable, AGILE_BUF_ARR &bufArr0, unsigned int offset, AgileLockChain &chain){
    if(enable){
        bufArr0.load(0, offset, chain);
    }
}

__device__ void wait_load(bool enable, AGILE_BUF_ARR &bufArr0){
    if(enable){
        bufArr0.wait();
    }
}


__global__ void ctc_sync_kernel(AGILE_CTRL * ctrl, AgileBuf * buf0,
    unsigned int buf_per_blk, unsigned int compute_sim, unsigned int compute_itr,
    unsigned int total_threads, unsigned int iteration,
    bool enable_load, bool enable_compute)
{
    AgileLockChain chain;
    __shared__ AGILE_BUF_ARR bufArr0;
    AGILE_BUF_ARR::init(ctrl, &bufArr0, buf0 + blockIdx.x * buf_per_blk, buf_per_blk);
    for(unsigned int i = 0; i < iteration; ++i){
        issue_load(enable_load, bufArr0, i * blockDim.x * buf_per_blk +
                                                blockIdx.x * buf_per_blk, chain);
        wait_load(enable_load, bufArr0);
        compute_krnl(enable_compute, bufArr0, compute_sim, compute_itr);
    }
}


__global__ void ctc_async_kernel(AGILE_CTRL * ctrl, AgileBuf * buf0, AgileBuf * buf1, unsigned int buf_per_blk,
    unsigned int compute_sim, unsigned int compute_itr, unsigned int total_threads, unsigned int iteration,
    bool enable_load, bool enable_compute)
{
    AgileLockChain chain;
    __shared__ AGILE_BUF_ARR bufArr0;
    AGILE_BUF_ARR::init(ctrl, &bufArr0, buf0 + blockIdx.x * buf_per_blk, buf_per_blk);
    __shared__ AGILE_BUF_ARR bufArr1;
    AGILE_BUF_ARR::init(ctrl, &bufArr1, buf1 + blockIdx.x * buf_per_blk, buf_per_blk);
    for(unsigned int i = 0; i < iteration + 1; ++i){
        if(i % 2 == 0){
            wait_load(enable_load && i > 0, bufArr1);
            issue_load(enable_load && i < iteration, bufArr0, i * blockDim.x * buf_per_blk + blockIdx.x * buf_per_blk, chain);
            compute_krnl(enable_compute && i > 0, bufArr1, compute_sim, compute_itr);
        }else{
            wait_load(enable_load && i > 0, bufArr0);
            issue_load(enable_load && i < iteration, bufArr1, i * blockDim.x * buf_per_blk + blockIdx.x * buf_per_blk, chain);
            compute_krnl(enable_compute && i > 0, bufArr0, compute_sim, compute_itr);
        }
    }
}


__global__ void resetCache(AGILE_CTRL * ctrl, AgileBuf * buf, unsigned int buf_per_blk, unsigned int total_threads){
    unsigned int AGILE_BID = blockIdx.x;
    unsigned int tid = AGILE_BID * blockDim.x + threadIdx.x;

    if(tid < buf_per_blk){
        buf[AGILE_BID * buf_per_blk + tid].resetStatus();
    }

    for(unsigned int i = tid; i < ctrl->cache_hierarchy->gpu_cache->slot_num; i += total_threads){
        ctrl->cache_hierarchy->gpu_cache->cache_status[i] = AGILE_GPUCACHE_EMPTY;
        static_cast<SimpleGPUCache<CPU_CACHE_IMPL, WRITE_TABLE_IMPL> *>(ctrl->cache_hierarchy->gpu_cache)->tag_blk_id[i] = -1;
        static_cast<SimpleGPUCache<CPU_CACHE_IMPL, WRITE_TABLE_IMPL> *>(ctrl->cache_hierarchy->gpu_cache)->tag_dev_id[i] = -1;
    }
}


int main(int argc, char ** argv){
    CLI::App app{"AGILE CTC (Compute-Transfer Concurrency) Demo"};

    // IPC queue name (empty = standalone mode, no IPC)
    std::string ipc_queue;
    app.add_option("--ipc-queue", ipc_queue, "IPC message queue name (set by server)");

    // Agile config
    unsigned int slot_size = 512;
    unsigned int gpu_slot_num = 65536 * 8;

    // NVMe config
    std::string nvme_device = "/dev/AGILE-NVMe-0000:e4:00.0";
    unsigned int queue_num = 8;
    unsigned int queue_depth = 512;
    unsigned int ssd_blk_offset = 0;

    // Parallelism config
    unsigned int block_dim = 1;
    unsigned int thread_dim = 32;
    unsigned int agile_dim = 1;

    // CTC config
    unsigned int buf_per_blk = 4;
    unsigned int compute_sim = 65536;
    unsigned int compute_itr = 490;
    unsigned int iteration = 5000;

    app.add_option("-d,--dev", nvme_device, "NVMe device path");
    app.add_option("--slot-size", slot_size, "Slot size of the cache");
    app.add_option("--gpu-slots", gpu_slot_num, "Number of GPU cache slots");
    app.add_option("-q,--queues", queue_num, "Number of NVMe queue pairs");
    app.add_option("--queue-depth", queue_depth, "Depth of each NVMe queue");
    app.add_option("--blk-offset", ssd_blk_offset, "SSD block offset");
    app.add_option("-b,--blocks", block_dim, "Block dimension");
    app.add_option("-t,--threads", thread_dim, "Thread dimension");
    app.add_option("--agile-dim", agile_dim, "Agile service blocks");
    app.add_option("--buf-per-blk", buf_per_blk, "Buffers per block");
    app.add_option("--compute-sim", compute_sim, "Compute simulation multiplier");
    app.add_option("--compute-itr", compute_itr, "Compute iterations per load");
    app.add_option("-i,--iterations", iteration, "Total iterations");

    CLI11_PARSE(app, argc, argv);

    if(iteration <= 1){
        LOG_ERROR("CTC", "Iteration should be greater than 1");
        return 1;
    }

    AGILE_HOST host(0, slot_size);

    CPU_CACHE_IMPL c_cache(0, slot_size);
    WRITE_TABLE_IMPL w_table(0);
    GPU_CACHE_IMPL g_cache(gpu_slot_num, slot_size);

    host.setGPUCache(g_cache);
    host.setCPUCache(c_cache);
    host.setShareTable(w_table);

    host.addNvmeDev(nvme_device, ssd_blk_offset, queue_num, queue_depth);
    host.initNvme();

    AgileBuf * buf0, * buf1, * buf2;
    LOG_INFO("CTC", "block dim: %d buf per blk: %d", block_dim, buf_per_blk);
    host.allocateBuffer(buf0, block_dim * buf_per_blk);
    host.allocateBuffer(buf1, block_dim * buf_per_blk);
    host.allocateBuffer(buf2, block_dim * buf_per_blk);

    host.configParallelism(block_dim, thread_dim, agile_dim);
    host.initializeAgile();

    auto *ctrl = host.getAgileCtrlDevicePtr();

    // We need to query occupancy before starting AGILE to make sure the kernel can be launched successfully and be executed in parallel with AGILE service. 
    host.queryOccupancy(ctc_sync_kernel, 1024, 0);
    host.queryOccupancy(ctc_async_kernel, 1024, 0);
    host.queryOccupancy(resetCache, 1024, 0);

    host.startAgile();

    std::chrono::high_resolution_clock::time_point start, end;
    long load_only_us, compute_only_us, sync_us, async_us;

    // 1. Compute only (no load)
    LOG_INFO("CTC", "=== Test 1: Compute Only ===");
    host.runKernel(resetCache, ctrl, buf0, buf_per_blk, block_dim * thread_dim);
    start = std::chrono::high_resolution_clock::now();
    host.runKernel(ctc_sync_kernel, ctrl, buf0, buf_per_blk, compute_sim, compute_itr, block_dim * thread_dim, iteration, false, true);
    end = std::chrono::high_resolution_clock::now();
    compute_only_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    LOG_INFO("CTC", "Compute only time: %ld us", compute_only_us);

    // 2. Load only (no compute)
    LOG_INFO("CTC", "=== Test 2: Load Only ===");
    // host.runKernel(resetCache, ctrl, buf0, buf_per_blk, block_dim * thread_dim);
    start = std::chrono::high_resolution_clock::now();
    host.runKernel(ctc_sync_kernel, ctrl, buf0, buf_per_blk, compute_sim, compute_itr, block_dim * thread_dim, iteration, true, false);
    end = std::chrono::high_resolution_clock::now();
    load_only_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    LOG_INFO("CTC", "Load only time: %ld us", load_only_us);
    
    // 3. Sync load + compute
    LOG_INFO("CTC", "=== Test 3: Sync Load + Compute ===");
    host.runKernel(resetCache, ctrl, buf0, buf_per_blk, block_dim * thread_dim);
    start = std::chrono::high_resolution_clock::now();
    host.runKernel(ctc_sync_kernel, ctrl, buf0, buf_per_blk, compute_sim, compute_itr, block_dim * thread_dim, iteration, true, true);
    end = std::chrono::high_resolution_clock::now();
    sync_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    LOG_INFO("CTC", "Sync time: %ld us", sync_us);

    // 4. Async load + compute (double-buffered)
    LOG_INFO("CTC", "=== Test 4: Async Load + Compute ===");
    host.runKernel(resetCache, ctrl, buf0, buf_per_blk, block_dim * thread_dim);
    start = std::chrono::high_resolution_clock::now();
    host.runKernel(ctc_async_kernel, ctrl, buf1, buf2, buf_per_blk, compute_sim, compute_itr, block_dim * thread_dim, iteration, true, true);
    end = std::chrono::high_resolution_clock::now();
    async_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    LOG_INFO("CTC", "Async time: %ld us", async_us);

    LOG_INFO("CTC", "=== Summary ===");
    LOG_INFO("CTC", "Compute only: %ld us", compute_only_us);
    LOG_INFO("CTC", "Load only:    %ld us", load_only_us);
    LOG_INFO("CTC", "Sync (L+C):   %ld us", sync_us);
    LOG_INFO("CTC", "Async (L+C):  %ld us", async_us);
    LOG_INFO("CTC", "CTC (Compute-to-Communication): %.4f", (double)compute_only_us / (double)load_only_us);
    LOG_INFO("CTC", "Async speedup over sync: %.4fx", (double)sync_us / (double)async_us);

    LOG_INFO("CTC", "kernel finish");
    host.stopAgile();

    host.freeBuffer(buf0, block_dim * buf_per_blk);
    host.freeBuffer(buf1, block_dim * buf_per_blk);
    host.freeBuffer(buf2, block_dim * buf_per_blk);
    host.closeNvme();

    /* ── Send results via IPC (if launched by the server) ─────────── */
    if (!ipc_queue.empty()) {
        demo_ipc::QueueClient ipc(ipc_queue);
        std::ostringstream os;
        os << "{" << "\"compute_only_us\": " << compute_only_us << ", "
           << "\"load_only_us\": " << load_only_us << ", "
           << "\"sync_us\": " << sync_us << ", "
           << "\"async_us\": " << async_us << ", "
           << "\"ctc_ratio\": " << (double)compute_only_us / (double)load_only_us << ", "
           << "\"async_speedup\": " << (double)sync_us / (double)async_us
           << "}";
        ipc.send(os.str());
        ipc.send_done();
    }

    return 0;
}
