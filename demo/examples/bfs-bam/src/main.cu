/*
 * BFS demo using the BAM (BaM) library for GPU-direct SSD access.
 *
 * This mirrors the baseline at baseline/benchmarks/test_bfs but uses
 * CLI11, the demo IPC layer, and info-file auto-resolution so it fits
 * into the AGILE demo framework.
 *
 * BAM requires its own libnvm kernel module loaded and /dev/libnvm*
 * device nodes present.
 */

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <CLI/CLI.hpp>

/* BAM headers */
#include <nvm_ctrl.h>
#include <nvm_types.h>
#include <nvm_queue.h>
#include <nvm_util.h>
#include <nvm_admin.h>
#include <nvm_error.h>
#include <nvm_cmd.h>
#include <ctrl.h>
#include <buffer.h>
#include <event.h>
#include <queue.h>
#include <nvm_parallel_queue.h>
#include <nvm_io.h>
#include <page_cache.h>
#include <util.h>

#include "demo_ipc.h"
#include <logger.hpp>

#define MYINFINITY 0xFFFFFFFF

typedef uint32_t EdgeT;

/* ── BFS kernel (BAM version) ─────────────────────────────── */
__global__
void bfs_kernel_bam(array_d_t<uint32_t> *da,
                    uint32_t *label, const uint32_t level,
                    const uint32_t vertex_count,
                    const uint32_t *vertexList,
                    uint32_t *changed) {
    const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < vertex_count && label[tid] == level) {
        const uint32_t start = vertexList[tid];
        const uint32_t end   = vertexList[tid + 1];

        for (uint32_t i = start; i < end; i++) {
            EdgeT next = da->seq_read(i);
            if (label[next] == MYINFINITY) {
                label[next] = level + 1;
                *changed = 1;
            }
        }
    }
}

/* ── Parse graph-gen .info.txt ────────────────────────────── */
static std::unordered_map<std::string, std::string>
parse_info(const std::string &path) {
    std::unordered_map<std::string, std::string> kv;
    std::ifstream in(path);
    if (!in) {
        LOG_ERROR("BFS-BAM", "Cannot open info file: %s", path.c_str());
        std::exit(1);
    }
    std::string line;
    while (std::getline(in, line)) {
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        while (!key.empty() && key.back() == ' ') key.pop_back();
        while (!val.empty() && val.front() == ' ') val.erase(val.begin());
        kv[key] = val;
    }
    return kv;
}

/* ── Read offsets (supports 4 or 8-byte on disk) ──────────── */
static std::vector<uint32_t>
read_offsets_file(const std::string &path, uint64_t count,
                  unsigned int offset_size) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        LOG_ERROR("BFS-BAM", "Cannot open offsets file: %s", path.c_str());
        std::exit(1);
    }
    std::vector<uint32_t> offsets(count);
    if (offset_size == 4) {
        in.read(reinterpret_cast<char *>(offsets.data()),
                count * sizeof(uint32_t));
    } else if (offset_size == 8) {
        std::vector<uint64_t> tmp(count);
        in.read(reinterpret_cast<char *>(tmp.data()),
                count * sizeof(uint64_t));
        for (uint64_t i = 0; i < count; ++i)
            offsets[i] = static_cast<uint32_t>(tmp[i]);
    } else {
        LOG_ERROR("BFS-BAM", "Unsupported offset_size: %u", offset_size);
        std::exit(1);
    }
    return offsets;
}


int main(int argc, char **argv) {
    CLI::App app{"BAM BFS (Breadth-First Search) Demo"};

    /* IPC */
    std::string ipc_queue;
    app.add_option("--ipc-queue", ipc_queue,
                   "IPC message queue name (set by server)");

    /* Info file */
    std::string info_file;
    app.add_option("-i,--info", info_file,
                   "Path to the .info.txt produced by graph-gen");

    /* BAM config */
    std::string nvme_device = "/dev/libnvm0";
    unsigned int page_size  = 4096;
    uint64_t     num_pages  = 65536 * 8;
    unsigned int queue_num  = 15;
    unsigned int queue_depth = 512;
    unsigned int gpu_id     = 0;
    unsigned int ns_id      = 1;

    /* Parallelism */
    unsigned int thread_dim = 1024;

    /* Graph config */
    unsigned int node_num   = 0;
    uint64_t     edge_num   = 0;
    unsigned int start_node = 0;
    std::string  offset_file;
    std::string  output_file = "res-bfs-bam.bin";

    app.add_option("-d,--dev", nvme_device, "BAM NVMe device (e.g. /dev/libnvm0)");
    app.add_option("--page-size", page_size, "Page (slot) size in bytes");
    app.add_option("--pages", num_pages, "Number of cache pages");
    app.add_option("-q,--queues", queue_num, "Number of NVMe queues");
    app.add_option("--queue-depth", queue_depth, "Queue depth");
    app.add_option("-g,--gpu", gpu_id, "CUDA device ID");
    app.add_option("--ns", ns_id, "NVMe namespace ID");
    app.add_option("-t,--threads", thread_dim, "Threads per block");
    app.add_option("--nodes", node_num, "Number of nodes");
    app.add_option("--edges", edge_num, "Number of edges (directed)");
    app.add_option("-s,--start", start_node, "Start node for BFS");
    app.add_option("--offset-file", offset_file, "Offsets binary file");
    app.add_option("-o,--output", output_file, "Output BFS result file");

    CLI11_PARSE(app, argc, argv);

    /* ── Resolve from info file ── */
    unsigned int offset_size = 4;
    if (!info_file.empty()) {
        auto kv = parse_info(info_file);
        auto get = [&](const std::string &key) -> std::string {
            auto it = kv.find(key);
            if (it == kv.end()) {
                LOG_ERROR("BFS-BAM", "Missing key '%s' in %s",
                          key.c_str(), info_file.c_str());
                std::exit(1);
            }
            return it->second;
        };
        if (offset_file.empty()) offset_file = get("offsets_file");
        /* Resolve relative to the info file's directory */
        if (!offset_file.empty() && offset_file[0] != '/') {
            auto info_dir = std::filesystem::path(info_file).parent_path();
            if (!info_dir.empty())
                offset_file = (info_dir / offset_file).string();
        }
        if (node_num == 0) node_num = std::stoul(get("num_nodes"));
        if (edge_num == 0) {
            auto it = kv.find("num_edges_directed");
            edge_num = (it != kv.end()) ? std::stoull(it->second)
                                        : std::stoull(get("num_edges"));
        }
        offset_size = std::stoul(get("offset_size"));
    }

    if (offset_file.empty() || node_num == 0) {
        LOG_ERROR("BFS-BAM", "Provide --info or --offset-file and --nodes");
        return 1;
    }
    if (start_node >= node_num) {
        LOG_ERROR("BFS-BAM", "start_node %u >= node_num %u", start_node, node_num);
        return 1;
    }

    LOG_INFO("BFS-BAM", "nodes: %u, edges(directed): %lu, start: %u",
             node_num, (unsigned long)edge_num, start_node);
    LOG_INFO("BFS-BAM", "BAM device: %s, pages: %lu, page_size: %u, queues: %u",
             nvme_device.c_str(), (unsigned long)num_pages, page_size, queue_num);

    /* ── Load offsets ── */
    LOG_INFO("BFS-BAM", "Loading offsets...");
    auto h_offsets_vec = read_offsets_file(offset_file, node_num + 1, offset_size);

    /* ── Setup BAM ── */
    cuda_err_chk(cudaSetDevice(gpu_id));

    Controller ctrl(nvme_device.c_str(), ns_id, gpu_id, queue_depth, queue_num);
    std::vector<Controller *> ctrls_vec = {&ctrl};

    page_cache_t h_pc(page_size, num_pages, gpu_id, ctrl, 64, ctrls_vec);

    uint64_t t_size = edge_num * sizeof(uint32_t);
    range_t<uint32_t> h_range(0, edge_num, 0,
                              (t_size / page_size) + 1,
                              0, page_size, &h_pc, gpu_id);

    std::vector<range_t<uint32_t> *> vr = {&h_range};
    array_t<uint32_t> nvme_mem(edge_num, 0, vr, gpu_id);

    LOG_INFO("BFS-BAM", "BAM initialized");

    /* ── Allocate device arrays ── */
    uint32_t *d_label, *d_offsets, *d_changed;
    uint32_t *h_label = (uint32_t *)malloc(node_num * sizeof(uint32_t));
    memset(h_label, 0xFF, node_num * sizeof(uint32_t));
    h_label[start_node] = 0;

    cuda_err_chk(cudaMalloc(&d_label,   node_num * sizeof(uint32_t)));
    cuda_err_chk(cudaMalloc(&d_offsets,  (node_num + 1) * sizeof(uint32_t)));
    cuda_err_chk(cudaMalloc(&d_changed,  sizeof(uint32_t)));

    cuda_err_chk(cudaMemcpy(d_label, h_label,
                            node_num * sizeof(uint32_t),
                            cudaMemcpyHostToDevice));
    cuda_err_chk(cudaMemcpy(d_offsets, h_offsets_vec.data(),
                            (node_num + 1) * sizeof(uint32_t),
                            cudaMemcpyHostToDevice));

    unsigned int grid_dim = node_num / thread_dim + 1;
    LOG_INFO("BFS-BAM", "Starting BFS... grid=%u threads=%u", grid_dim, thread_dim);

    /* ── BFS Pass 1: COLD — data loaded from SSD into page cache ── */
    double cold_itr_time = 0;
    unsigned int cold_level = 0;
    uint32_t changed = 0;

    LOG_INFO("BFS-BAM", "=== Pass 1: COLD (loading from SSD) ===");
    auto t_start = std::chrono::high_resolution_clock::now();
    do {
        changed = 0;
        cuda_err_chk(cudaMemcpy(d_changed, &changed, sizeof(uint32_t),
                                cudaMemcpyHostToDevice));

        auto s0 = std::chrono::high_resolution_clock::now();
        bfs_kernel_bam<<<grid_dim, thread_dim>>>(
            nvme_mem.d_array_ptr, d_label, cold_level,
            node_num, d_offsets, d_changed);
        cuda_err_chk(cudaDeviceSynchronize());
        auto e0 = std::chrono::high_resolution_clock::now();

        double itr_time =
            std::chrono::duration_cast<std::chrono::duration<double>>(e0 - s0)
                .count();
        cold_itr_time += itr_time;

        cuda_err_chk(cudaMemcpy(&changed, d_changed, sizeof(uint32_t),
                                cudaMemcpyDeviceToHost));
        LOG_INFO("BFS-BAM", "[COLD] level %u  changed=%u  time=%.6f s",
                 cold_level, changed, itr_time);
        cold_level++;
    } while (changed);
    auto t_end = std::chrono::high_resolution_clock::now();
    double cold_total_time =
        std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start)
            .count();
    LOG_INFO("BFS-BAM", "COLD done: %u levels, kernel time=%.6f s, total=%.6f s",
             cold_level, cold_itr_time, cold_total_time);

    /* ── BFS Pass 2: WARM — data already in page cache ── */
    // Reset labels
    cuda_err_chk(cudaMemcpy(d_label, h_label,
                            node_num * sizeof(uint32_t),
                            cudaMemcpyHostToDevice));

    double warm_itr_time = 0;
    unsigned int warm_level = 0;

    LOG_INFO("BFS-BAM", "=== Pass 2: WARM (data in page cache) ===");
    t_start = std::chrono::high_resolution_clock::now();
    do {
        changed = 0;
        cuda_err_chk(cudaMemcpy(d_changed, &changed, sizeof(uint32_t),
                                cudaMemcpyHostToDevice));

        auto s0 = std::chrono::high_resolution_clock::now();
        bfs_kernel_bam<<<grid_dim, thread_dim>>>(
            nvme_mem.d_array_ptr, d_label, warm_level,
            node_num, d_offsets, d_changed);
        cuda_err_chk(cudaDeviceSynchronize());
        auto e0 = std::chrono::high_resolution_clock::now();

        double itr_time =
            std::chrono::duration_cast<std::chrono::duration<double>>(e0 - s0)
                .count();
        warm_itr_time += itr_time;

        cuda_err_chk(cudaMemcpy(&changed, d_changed, sizeof(uint32_t),
                                cudaMemcpyDeviceToHost));
        LOG_INFO("BFS-BAM", "[WARM] level %u  changed=%u  time=%.6f s",
                 warm_level, changed, itr_time);
        warm_level++;
    } while (changed);
    t_end = std::chrono::high_resolution_clock::now();
    double warm_total_time =
        std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start)
            .count();

    unsigned int level = warm_level;
    double total_itr_time = warm_itr_time;
    double total_time = warm_total_time;

    LOG_INFO("BFS-BAM", "WARM done: %u levels, kernel time=%.6f s, total=%.6f s",
             level, warm_itr_time, warm_total_time);
    LOG_INFO("BFS-BAM", "=== Summary ===");
    LOG_INFO("BFS-BAM", "  COLD kernel: %.6f s   WARM kernel: %.6f s   Speedup: %.2fx",
             cold_itr_time, warm_itr_time, cold_itr_time / warm_itr_time);

    /* ── Copy results back ── */
    cuda_err_chk(cudaMemcpy(h_label, d_label,
                            node_num * sizeof(uint32_t),
                            cudaMemcpyDeviceToHost));

    unsigned int visited = 0;
    for (unsigned int i = 0; i < node_num; i++) {
        if (h_label[i] != MYINFINITY) visited++;
    }
    LOG_INFO("BFS-BAM", "Visited: %u / %u nodes", visited, node_num);

    if (!output_file.empty()) {
        std::ofstream ofs(output_file, std::ios::binary);
        ofs.write(reinterpret_cast<const char *>(h_label),
                  node_num * sizeof(uint32_t));
        ofs.close();
        LOG_INFO("BFS-BAM", "Result saved to %s", output_file.c_str());
    }

    /* ── Cleanup ── */
    cuda_err_chk(cudaFree(d_label));
    cuda_err_chk(cudaFree(d_offsets));
    cuda_err_chk(cudaFree(d_changed));
    free(h_label);

    /* ── IPC ── */
    if (!ipc_queue.empty()) {
        demo_ipc::QueueClient ipc(ipc_queue);
        ipc.send("=== BFS-BAM (Breadth-First Search, BAM baseline) ===");
        { std::ostringstream os; os << "Graph: " << node_num << " nodes, "
              << edge_num << " edges (directed)"; ipc.send(os.str()); }
        { std::ostringstream os; os << "Start node: " << start_node;
              ipc.send(os.str()); }
        { std::ostringstream os; os << "Levels: " << level;
              ipc.send(os.str()); }
        { std::ostringstream os; os << "Visited: " << visited << " / "
              << node_num << " nodes"; ipc.send(os.str()); }
        { std::ostringstream os; os << "COLD kernel time: " << cold_itr_time
              << " s"; ipc.send(os.str()); }
        { std::ostringstream os; os << "WARM kernel time: " << warm_itr_time
              << " s"; ipc.send(os.str()); }
        { std::ostringstream os; os << "Speedup (cold/warm): "
              << (cold_itr_time / warm_itr_time) << "x"; ipc.send(os.str()); }
        ipc.send_done();
    }

    return 0;
}
