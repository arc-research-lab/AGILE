#include <iostream>
#include <fstream>
#include <cstdio>
#include <string>
#include <sstream>

#include <CLI/CLI.hpp>

#include "agile_host.h"
#include "cache_impl.h"
#include "table_impl.h"
#include "demo_ipc.h"



#define CPU_CACHE_IMPL DisableCPUCache
#define SHARE_TABLE_IMPL SimpleShareTable
// #define GPU_CACHE_IMPL SimpleGPUCache<CPU_CACHE_IMPL, SHARE_TABLE_IMPL>
#define GPU_CACHE_IMPL GPUClockReplacementCache<CPU_CACHE_IMPL, SHARE_TABLE_IMPL>

#define AGILE_CTRL AgileCtrl<GPU_CACHE_IMPL, CPU_CACHE_IMPL, SHARE_TABLE_IMPL>
#define AGILE_HOST AgileHost<GPU_CACHE_IMPL, CPU_CACHE_IMPL, SHARE_TABLE_IMPL>


__global__ void gpu_kernel(AGILE_CTRL * ctrl, unsigned int length){
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    AgileLockChain chain;
    if(tid < length){
        auto agileArr = ctrl->getArrayWrap<unsigned int>(chain);
        printf("Thread %d read: %d\n", tid, agileArr[0][tid]);
    }
}

int main(int argc, char ** argv){
    CLI::App app{"AGILE Hello World Demo"};

    // IPC queue name (empty = standalone mode, no IPC)
    std::string ipc_queue;
    app.add_option("--ipc-queue", ipc_queue, "IPC message queue name (set by server)");

    // Agile config
    unsigned int slot_size = 4096;
    unsigned int gpu_slot_num = 65536 * 8;

    // NVMe config
    std::string nvme_device = "/dev/AGILE-NVMe-0000:01:00.0";
    unsigned int queue_num = 32;
    unsigned int queue_depth = 256;
    unsigned int ssd_blk_offset = 0;
    unsigned int ssd_block_num = 1048576;

    app.add_option("-d,--dev", nvme_device, "NVMe device path");
    app.add_option("--slot-size", slot_size, "Slot size of the cache");
    app.add_option("--gpu-slots", gpu_slot_num, "Number of GPU cache slots");
    app.add_option("-q,--queues", queue_num, "Number of NVMe queue pairs");
    app.add_option("--queue-depth", queue_depth, "Depth of each NVMe queue");
    app.add_option("--blk-offset", ssd_blk_offset, "SSD block offset");
    app.add_option("--ssd-blocks", ssd_block_num, "Number of SSD blocks");

    CLI11_PARSE(app, argc, argv);

    AGILE_HOST host(0, slot_size);

    CPU_CACHE_IMPL c_cache(0, slot_size);
    SHARE_TABLE_IMPL w_table(gpu_slot_num / 4);
    GPU_CACHE_IMPL g_cache(gpu_slot_num, slot_size, ssd_block_num);

    host.setGPUCache(g_cache);
    host.setCPUCache(c_cache);
    host.setShareTable(w_table);

    host.addNvmeDev(nvme_device, ssd_blk_offset, queue_num, queue_depth);
    host.initNvme();

    host.initializeAgile();

    // We need to query occupancy before starting AGILE to make sure the kernel can be launched successfully and executed in parallel with AGILE service. 
    host.queryOccupancy(gpu_kernel, 1024, 0);

    host.startAgile();

    cudaStream_t krnl_stream;
    cudaStreamCreate(&krnl_stream);

    gpu_kernel<<<1, 1024, 0, krnl_stream>>>(host.getAgileCtrlDevicePtr(), 8);
    
    cuda_err_chk(cudaStreamSynchronize(krnl_stream));
    cuda_err_chk(cudaStreamDestroy(krnl_stream));

    host.stopAgile();
    host.closeNvme();

    /* ── Send results via IPC (if launched by the server) ─────────── */
    if (!ipc_queue.empty()) {
        demo_ipc::QueueClient ipc(ipc_queue);
        ipc.send("Hello World demo completed successfully.");
        ipc.send("Initialized AGILE on GPU 0 with " +
                 std::to_string(gpu_slot_num) + " cache slots (slot size " +
                 std::to_string(slot_size) + " B).");
        ipc.send("NVMe device: " + nvme_device +
                 " | queues: " + std::to_string(queue_num) +
                 " | depth: " + std::to_string(queue_depth));
        ipc.send("Launched 8 GPU threads; each read data from NVMe-backed array.");
        ipc.send_done();
    }

    return 0;
}