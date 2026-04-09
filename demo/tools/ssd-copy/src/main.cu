#include <cstdio>
#include <string>
#include <chrono>
#include <sstream>

#include <CLI/CLI.hpp>
#include <logger.hpp>

#include "agile_host.h"
#include "cache_impl.h"
#include "table_impl.h"
#include "demo_ipc.h"

#define CPU_CACHE_IMPL  DisableCPUCache
#define SHARE_TABLE_IMPL DisableShareTable
#define GPU_CACHE_IMPL  SimpleGPUCache<CPU_CACHE_IMPL, SHARE_TABLE_IMPL>

#define AGILE_CTRL AgileCtrl<GPU_CACHE_IMPL, CPU_CACHE_IMPL, SHARE_TABLE_IMPL>
#define AGILE_HOST AgileHost<GPU_CACHE_IMPL, CPU_CACHE_IMPL, SHARE_TABLE_IMPL>


/* ── GPU kernels ──────────────────────────────────────────────── */

__global__ void read_kernel(AGILE_CTRL *ctrl, AgileBuf *buf,
                            unsigned int block_offset)
{
    AgileLockChain chain;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    AgileBufPtr bufPtr(buf[tid]);
    bufPtr.resetStatus();
    ctrl->asyncRead(0, block_offset + tid, bufPtr, chain);
    bufPtr.wait();
}

__global__ void write_kernel(AGILE_CTRL *ctrl, AgileBuf *buf,
                             unsigned int block_offset)
{
    AgileLockChain chain;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    AgileBufPtr bufPtr(buf[tid]);
    ctrl->writeThroughNvme(0, block_offset + tid, bufPtr, chain);
}


/* ── helpers ──────────────────────────────────────────────────── */

static void do_read(AGILE_HOST &host, const std::string &nvme_device,
                    unsigned int ssd_blk_offset, unsigned int queue_num,
                    unsigned int queue_depth, unsigned int slot_size,
                    unsigned int gpu_slot_num,
                    unsigned int block_dim, unsigned int thread_dim,
                    unsigned int agile_dim, unsigned int ssd_blk_num,
                    const std::string &output_file,
                    const std::string &ipc_queue)
{
    host.addNvmeDev(nvme_device, ssd_blk_offset, queue_num, queue_depth);
    host.initNvme();

    AgileBuf *buf;
    unsigned int batch = block_dim * thread_dim;

    host.allocateBuffer(buf, batch);
    host.configParallelism(block_dim, thread_dim, agile_dim);
    host.initializeAgile();

    host.queryOccupancy(read_kernel, thread_dim, 0);

    remove(output_file.c_str());
    auto *ctrl = host.getAgileCtrlDevicePtr();
    host.startAgile();

    auto t0 = std::chrono::high_resolution_clock::now();

    for (unsigned long i = 0; i < ssd_blk_num; i += batch) {
        unsigned int remaining = ssd_blk_num - i;
        unsigned int count = (remaining < batch) ? remaining : batch;
        LOG_INFO("SSD-COPY", "read progress: %lu / %u", i, ssd_blk_num);
        host.runKernel(read_kernel, ctrl, buf, (unsigned int)i);
        host.appendBuf2File(output_file, buf, count);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    long elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    double elapsed_s = elapsed_us / 1.0e6;
    double total_bytes = (double)ssd_blk_num * slot_size;
    double throughput_mbs = total_bytes / elapsed_s / (1024.0 * 1024.0);

    LOG_INFO("SSD-COPY", "Read complete: %u blocks (%u B each) in %.3f s  (%.2f MB/s)",
             ssd_blk_num, slot_size, elapsed_s, throughput_mbs);
    LOG_INFO("SSD-COPY", "Output file: %s", output_file.c_str());

    host.stopAgile();
    host.freeBuffer(buf, batch);
    host.closeNvme();

    if (!ipc_queue.empty()) {
        demo_ipc::QueueClient ipc(ipc_queue);
        {
            std::ostringstream os;
            os << "Read " << ssd_blk_num << " blocks (" << slot_size
               << " B each) in " << elapsed_s << " s";
            ipc.send(os.str());
        }
        {
            std::ostringstream os;
            os << "Throughput: " << throughput_mbs << " MB/s";
            ipc.send(os.str());
        }
        ipc.send("Output file: " + output_file);
        ipc.send_done();
    }
}

static void do_write(AGILE_HOST &host, const std::string &nvme_device,
                     unsigned int ssd_blk_offset, unsigned int queue_num,
                     unsigned int queue_depth, unsigned int slot_size,
                     unsigned int gpu_slot_num,
                     unsigned int block_dim, unsigned int thread_dim,
                     unsigned int agile_dim, unsigned int ssd_blk_num,
                     const std::string &input_file,
                     const std::string &ipc_queue)
{
    host.addNvmeDev(nvme_device, ssd_blk_offset, queue_num, queue_depth);
    host.initNvme();

    AgileBuf *buf;
    unsigned int batch = block_dim * thread_dim;

    host.allocateBuffer(buf, batch);
    host.configParallelism(block_dim, thread_dim, agile_dim);
    host.initializeAgile();

    host.queryOccupancy(write_kernel, thread_dim, 0);

    auto *ctrl = host.getAgileCtrlDevicePtr();
    host.startAgile();

    auto t0 = std::chrono::high_resolution_clock::now();

    for (unsigned long i = 0; i < ssd_blk_num; i += batch) {
        LOG_INFO("SSD-COPY", "write progress: %lu / %u", i, ssd_blk_num);
        host.loadFile2Buf(input_file, i, buf, batch);
        host.runKernel(write_kernel, ctrl, buf, (unsigned int)i);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    long elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    double elapsed_s = elapsed_us / 1.0e6;
    double total_bytes = (double)ssd_blk_num * slot_size;
    double throughput_mbs = total_bytes / elapsed_s / (1024.0 * 1024.0);

    LOG_INFO("SSD-COPY", "Write complete: %u blocks (%u B each) in %.3f s  (%.2f MB/s)",
             ssd_blk_num, slot_size, elapsed_s, throughput_mbs);
    LOG_INFO("SSD-COPY", "Input file: %s", input_file.c_str());

    host.stopAgile();
    host.freeBuffer(buf, batch);
    host.closeNvme();

    if (!ipc_queue.empty()) {
        demo_ipc::QueueClient ipc(ipc_queue);
        {
            std::ostringstream os;
            os << "Wrote " << ssd_blk_num << " blocks (" << slot_size
               << " B each) in " << elapsed_s << " s";
            ipc.send(os.str());
        }
        {
            std::ostringstream os;
            os << "Throughput: " << throughput_mbs << " MB/s";
            ipc.send(os.str());
        }
        ipc.send("Input file: " + input_file);
        ipc.send_done();
    }
}


/* ── main ─────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    CLI::App app{"AGILE SSD Copy Tool — read SSD to file  /  write file to SSD"};
    app.require_subcommand(1);

    /* ── shared options ── */
    std::string nvme_device = "/dev/AGILE-NVMe-0000:01:00.0";
    unsigned int slot_size     = 4096;
    unsigned int gpu_slot_num  = 1024;
    unsigned int queue_num     = 32;
    unsigned int queue_depth   = 512;
    unsigned int ssd_blk_offset = 0;
    unsigned int ssd_blk_num   = 262144;  // e.g. 1 GB at 4096 B slots
    unsigned int block_dim     = 1;
    unsigned int thread_dim    = 1024;
    unsigned int agile_dim     = 1;
    std::string ipc_queue;

    auto add_common = [&](CLI::App *sub) {
        sub->add_option("-d,--dev",        nvme_device,    "NVMe device path");
        sub->add_option("--slot-size",     slot_size,      "Slot (block) size in bytes");
        sub->add_option("--gpu-slots",     gpu_slot_num,   "Number of GPU cache slots");
        sub->add_option("-q,--queues",     queue_num,      "Number of NVMe queue pairs");
        sub->add_option("--queue-depth",   queue_depth,    "Depth of each NVMe queue");
        sub->add_option("--blk-offset",    ssd_blk_offset, "SSD starting block offset");
        sub->add_option("-n,--num-blocks", ssd_blk_num,    "Number of SSD blocks to transfer");
        sub->add_option("-b,--blocks",     block_dim,      "CUDA grid block dimension");
        sub->add_option("-t,--threads",    thread_dim,     "CUDA block thread dimension");
        sub->add_option("--agile-dim",     agile_dim,      "AGILE service blocks");
        sub->add_option("--ipc-queue",     ipc_queue,      "IPC queue name (set by server)");
    };

    /* ── read subcommand ── */
    std::string output_file = "output.bin";
    auto *read_cmd = app.add_subcommand("read", "Read SSD blocks to a file");
    add_common(read_cmd);
    read_cmd->add_option("-o,--output", output_file, "Output file path");

    /* ── write subcommand ── */
    std::string input_file;
    auto *write_cmd = app.add_subcommand("write", "Write a file to SSD blocks");
    add_common(write_cmd);
    write_cmd->add_option("-i,--input", input_file, "Input file path")->required();

    CLI11_PARSE(app, argc, argv);

    AGILE_HOST host(0, slot_size);

    CPU_CACHE_IMPL  c_cache(0, slot_size);
    SHARE_TABLE_IMPL w_table(0);
    GPU_CACHE_IMPL  g_cache(gpu_slot_num, slot_size);

    host.setGPUCache(g_cache);
    host.setCPUCache(c_cache);
    host.setShareTable(w_table);

    if (read_cmd->parsed()) {
        LOG_INFO("SSD-COPY", "Mode: READ  |  dev=%s  blocks=%u  slot=%u B  output=%s",
                 nvme_device.c_str(), ssd_blk_num, slot_size, output_file.c_str());
        do_read(host, nvme_device, ssd_blk_offset, queue_num, queue_depth,
                slot_size, gpu_slot_num, block_dim, thread_dim, agile_dim,
                ssd_blk_num, output_file, ipc_queue);
    } else if (write_cmd->parsed()) {
        LOG_INFO("SSD-COPY", "Mode: WRITE  |  dev=%s  blocks=%u  slot=%u B  input=%s",
                 nvme_device.c_str(), ssd_blk_num, slot_size, input_file.c_str());
        do_write(host, nvme_device, ssd_blk_offset, queue_num, queue_depth,
                 slot_size, gpu_slot_num, block_dim, thread_dim, agile_dim,
                 ssd_blk_num, input_file, ipc_queue);
    }

    return 0;
}
