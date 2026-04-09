/*
 * AGILE PageRank demo — iterative SpMV on SSD-backed CSR graph.
 *
 * SSD layout: [neighbors (col indices)] [normalised weights]
 *             ^offset 0                 ^weight_offset
 *
 * Runs BFS-style cold/warm comparison:
 *   Pass 1 (COLD): data fetched from SSD into software cache
 *   Pass 2 (WARM): data already resident in software cache
 */

#include <chrono>
#include <cmath>
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

#include "agile_host.h"
#include "cache_impl.h"
#include "table_impl.h"
#include "demo_ipc.h"
#include <logger.hpp>

#define CPU_CACHE_IMPL DisableCPUCache
#define SHARE_TABLE_IMPL SimpleShareTable
#define GPU_CACHE_IMPL GPUClockReplacementCache<CPU_CACHE_IMPL, SHARE_TABLE_IMPL>

#define AGILE_CTRL AgileCtrl<GPU_CACHE_IMPL, CPU_CACHE_IMPL, SHARE_TABLE_IMPL>
#define AGILE_HOST AgileHost<GPU_CACHE_IMPL, CPU_CACHE_IMPL, SHARE_TABLE_IMPL>

/* ── PageRank SpMV kernel ──────────────────────────────────── */
__global__ void pagerank_kernel(AGILE_CTRL *ctrl,
                                unsigned int *offsets,
                                unsigned int weight_offset,
                                float damping,
                                float *vec,
                                float *output_vec,
                                float *norm2,
                                unsigned int nodes,
                                unsigned int thread_num) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nodes) return;

    AgileLockChain chain;
    output_vec[tid] = 0;
    auto agileArr = ctrl->getArrayWrap<unsigned int>(chain);
    unsigned int start = offsets[tid];
    unsigned int end   = offsets[tid + 1];
    for (unsigned int j = start; j < end; ++j) {
        unsigned int col = agileArr[0][j];
        unsigned int val = agileArr[0][weight_offset + j];
        output_vec[tid] += __uint_as_float(val) * vec[col];
    }
    output_vec[tid] *= damping;
    output_vec[tid] += (1.0f - damping) / nodes;
    float diff = output_vec[tid] - vec[tid];
    atomicAdd(norm2, diff * diff);
}

/* ── Parse graph-gen .info.txt ─────────────────────────────── */
static std::unordered_map<std::string, std::string>
parse_info(const std::string &path) {
    std::unordered_map<std::string, std::string> kv;
    std::ifstream in(path);
    if (!in) { LOG_ERROR("PR", "Cannot open info file: %s", path.c_str()); std::exit(1); }
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

/* ── Read offsets (4 or 8 bytes → uint32) ──────────────────── */
static std::vector<unsigned int>
read_offsets_file(const std::string &path, uint64_t count,
                  unsigned int offset_size) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { LOG_ERROR("PR", "Cannot open offsets file: %s", path.c_str()); std::exit(1); }
    std::vector<unsigned int> offsets(count);
    if (offset_size == 4) {
        in.read(reinterpret_cast<char *>(offsets.data()), count * sizeof(uint32_t));
    } else if (offset_size == 8) {
        std::vector<uint64_t> tmp(count);
        in.read(reinterpret_cast<char *>(tmp.data()), count * sizeof(uint64_t));
        for (uint64_t i = 0; i < count; ++i)
            offsets[i] = static_cast<unsigned int>(tmp[i]);
    } else {
        LOG_ERROR("PR", "Unsupported offset_size: %u", offset_size);
        std::exit(1);
    }
    return offsets;
}


int main(int argc, char **argv) {
    CLI::App app{"AGILE PageRank Demo"};

    std::string ipc_queue;
    app.add_option("--ipc-queue", ipc_queue, "IPC message queue name");

    std::string info_file;
    app.add_option("-i,--info", info_file, "Path to .info.txt");

    /* AGILE config */
    unsigned int slot_size = 4096;
    unsigned int gpu_slot_num = 65536 * 8;
    std::string nvme_device = "/dev/AGILE-NVMe-0000:01:00.0";
    unsigned int queue_num = 15;
    unsigned int queue_depth = 512;
    unsigned int ssd_blk_offset = 0;

    /* Parallelism */
    unsigned int block_dim = 32;
    unsigned int thread_dim = 1024;
    unsigned int agile_dim = 1;

    /* Graph config */
    unsigned int node_num = 0;
    unsigned int edge_num = 0;
    unsigned int ssd_block_num = 1048576;
    unsigned int weight_offset = 0;
    std::string offset_file;
    std::string output_file = "res-pr.bin";

    /* PageRank config */
    float damping = 0.85f;
    unsigned int max_itr = 20;
    float error_thresh = 1e-6f;

    app.add_option("-d,--dev", nvme_device, "NVMe device path");
    app.add_option("--slot-size", slot_size, "Slot size");
    app.add_option("--gpu-slots", gpu_slot_num, "GPU cache slots");
    app.add_option("-q,--queues", queue_num, "NVMe queue pairs");
    app.add_option("--queue-depth", queue_depth, "Queue depth");
    app.add_option("--blk-offset", ssd_blk_offset, "SSD block offset");
    app.add_option("-b,--blocks", block_dim, "Block dimension");
    app.add_option("-t,--threads", thread_dim, "Thread dimension");
    app.add_option("--agile-dim", agile_dim, "AGILE service blocks");
    app.add_option("--nodes", node_num, "Number of nodes");
    app.add_option("--edges", edge_num, "Number of edges (directed)");
    app.add_option("--ssd-blocks", ssd_block_num, "Number of SSD blocks");
    app.add_option("--weight-offset", weight_offset, "Weight offset in SSD array (uint32 elements)");
    app.add_option("--offset-file", offset_file, "Offsets binary file");
    app.add_option("-o,--output", output_file, "Output PR result file");
    app.add_option("--damping", damping, "Damping factor");
    app.add_option("--max-itr", max_itr, "Max iterations");
    app.add_option("--error", error_thresh, "Convergence threshold");

    CLI11_PARSE(app, argc, argv);

    /* ── Resolve from info file ── */
    unsigned int offset_size = 4;
    if (!info_file.empty()) {
        auto kv = parse_info(info_file);
        auto get = [&](const std::string &key) -> std::string {
            auto it = kv.find(key);
            if (it == kv.end()) { LOG_ERROR("PR", "Missing key '%s'", key.c_str()); std::exit(1); }
            return it->second;
        };

        if (offset_file.empty()) {
            offset_file = get("offsets_file");
            if (!offset_file.empty() && offset_file[0] != '/') {
                auto info_dir = std::filesystem::path(info_file).parent_path();
                if (!info_dir.empty()) offset_file = (info_dir / offset_file).string();
            }
        }
        if (node_num == 0) node_num = std::stoul(get("num_nodes"));
        if (edge_num == 0) {
            auto it = kv.find("num_edges_directed");
            edge_num = (it != kv.end()) ? std::stoul(it->second) : std::stoul(get("num_edges"));
        }
        offset_size = std::stoul(get("offset_size"));

        /* ssd_block_num: total SSD data = neighbors + weights */
        auto it_nf = kv.find("neighbors_filesize");
        if (it_nf != kv.end()) {
            uint64_t neighbors_filesize = std::stoull(it_nf->second);
            /* weight_offset = neighbors_filesize / sizeof(uint32_t) */
            if (weight_offset == 0)
                weight_offset = static_cast<unsigned int>(neighbors_filesize / sizeof(uint32_t));
            /* SSD holds neighbors + weights; weights file size = neighbors file size (same #edges) */
            auto it_wf = kv.find("weights_filesize");
            uint64_t weights_filesize = it_wf != kv.end() ? std::stoull(it_wf->second) : neighbors_filesize;
            ssd_block_num = static_cast<unsigned int>((neighbors_filesize + weights_filesize + slot_size - 1) / slot_size);
            LOG_INFO("PR", "Auto ssd_block_num=%u, weight_offset=%u", ssd_block_num, weight_offset);
        }
    }

    if (offset_file.empty() || node_num == 0) {
        LOG_ERROR("PR", "Provide --info or --offset-file and --nodes");
        return 1;
    }

    LOG_INFO("PR", "nodes: %u, edges(directed): %u", node_num, edge_num);
    LOG_INFO("PR", "damping: %.2f, max_itr: %u, error: %g", damping, max_itr, error_thresh);
    LOG_INFO("PR", "weight_offset: %u, ssd_blocks: %u", weight_offset, ssd_block_num);
    LOG_INFO("PR", "device: %s, queues: %u, gpu_slots: %u", nvme_device.c_str(), queue_num, gpu_slot_num);

    /* ── Load offsets ── */
    LOG_INFO("PR", "Loading offsets...");
    auto h_offsets_vec = read_offsets_file(offset_file, node_num + 1, offset_size);

    /* ── Setup AGILE ── */
    AGILE_HOST host(0, slot_size);
    CPU_CACHE_IMPL c_cache(0, slot_size);
    SHARE_TABLE_IMPL w_table(gpu_slot_num / 4);
    GPU_CACHE_IMPL g_cache(gpu_slot_num, slot_size, ssd_block_num);

    host.setGPUCache(g_cache);
    host.setCPUCache(c_cache);
    host.setShareTable(w_table);
    host.addNvmeDev(nvme_device, ssd_blk_offset, queue_num, queue_depth);
    host.initNvme();

    /* ── Allocate GPU arrays ── */
    unsigned int *d_offsets;
    float *d_vec, *d_out_vec, *d_norm2;
    cuda_err_chk(cudaMalloc(&d_offsets, (node_num + 1) * sizeof(unsigned int)));
    cuda_err_chk(cudaMalloc(&d_vec, node_num * sizeof(float)));
    cuda_err_chk(cudaMalloc(&d_out_vec, node_num * sizeof(float)));
    cuda_err_chk(cudaMalloc(&d_norm2, sizeof(float)));
    cuda_err_chk(cudaMemcpy(d_offsets, h_offsets_vec.data(),
                            (node_num + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice));

    float initial_val = 1.0f / node_num;
    std::vector<float> h_vec(node_num, initial_val);
    cuda_err_chk(cudaMemcpy(d_vec, h_vec.data(), node_num * sizeof(float), cudaMemcpyHostToDevice));

    unsigned int blockDim_val = node_num / thread_dim + 1;
    host.configParallelism(blockDim_val, thread_dim, agile_dim);
    host.queryOccupancy(pagerank_kernel, 1024, 0);
    host.initializeAgile();
    auto *ctrl = host.getAgileCtrlDevicePtr();

    /* ── Helper lambda: run PR iterations ── */
    auto run_pr = [&](const char *tag) -> std::pair<double, unsigned int> {
        /* Reset vec */
        cuda_err_chk(cudaMemcpy(d_vec, h_vec.data(), node_num * sizeof(float), cudaMemcpyHostToDevice));

        double total_kernel_time = 0;
        unsigned int itr = 0;
        float h_norm2 = 0;
        auto t0 = std::chrono::high_resolution_clock::now();
        do {
            h_norm2 = 0;
            cuda_err_chk(cudaMemcpy(d_norm2, &h_norm2, sizeof(float), cudaMemcpyHostToDevice));

            auto s0 = std::chrono::high_resolution_clock::now();
            host.runKernel(pagerank_kernel, ctrl, d_offsets, weight_offset,
                           damping, d_vec, d_out_vec, d_norm2, node_num,
                           block_dim * thread_dim);
            auto e0 = std::chrono::high_resolution_clock::now();

            cuda_err_chk(cudaMemcpy(d_vec, d_out_vec, node_num * sizeof(float), cudaMemcpyDeviceToDevice));
            cuda_err_chk(cudaMemcpy(&h_norm2, d_norm2, sizeof(float), cudaMemcpyDeviceToHost));
            h_norm2 = sqrtf(h_norm2);

            double itr_time = std::chrono::duration_cast<std::chrono::duration<double>>(e0 - s0).count();
            total_kernel_time += itr_time;
            LOG_INFO("PR", "[%s] itr %u  norm=%.6e  time=%.6f s", tag, itr, h_norm2, itr_time);
            itr++;
        } while (itr < max_itr && h_norm2 > error_thresh);
        auto t1 = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
        LOG_INFO("PR", "[%s] done: %u itr, kernel=%.6f s, total=%.6f s",
                 tag, itr, total_kernel_time, total_time);
        return {total_kernel_time, itr};
    };

    /* ── Pass 1: COLD ── */
    LOG_INFO("PR", "=== Pass 1: COLD (loading from SSD) ===");
    host.startAgile();
    auto [cold_time, cold_itr] = run_pr("COLD");

    /* ── Pass 2: WARM ── */
    LOG_INFO("PR", "=== Pass 2: WARM (data in software cache) ===");
    auto [warm_time, warm_itr] = run_pr("WARM");
    host.stopAgile();

    LOG_INFO("PR", "=== Summary ===");
    LOG_INFO("PR", "  COLD kernel: %.6f s   WARM kernel: %.6f s   Speedup: %.2fx",
             cold_time, warm_time, cold_time / warm_time);

    /* ── Save result ── */
    std::vector<float> h_result(node_num);
    cuda_err_chk(cudaMemcpy(h_result.data(), d_vec, node_num * sizeof(float), cudaMemcpyDeviceToHost));

    unsigned int nonzero = 0;
    float max_val = 0, min_val = 1;
    for (unsigned int i = 0; i < node_num; i++) {
        if (h_result[i] > max_val) max_val = h_result[i];
        if (h_result[i] < min_val) min_val = h_result[i];
        if (h_result[i] != initial_val) nonzero++;
    }
    LOG_INFO("PR", "min=%.6e  max=%.6e  changed=%u/%u", min_val, max_val, nonzero, node_num);

    if (!output_file.empty()) {
        std::ofstream ofs(output_file, std::ios::binary);
        ofs.write(reinterpret_cast<const char *>(h_result.data()), node_num * sizeof(float));
        ofs.close();
        LOG_INFO("PR", "Result saved to %s", output_file.c_str());
    }

    /* ── Cleanup ── */
    cuda_err_chk(cudaFree(d_offsets));
    cuda_err_chk(cudaFree(d_vec));
    cuda_err_chk(cudaFree(d_out_vec));
    cuda_err_chk(cudaFree(d_norm2));
    host.closeNvme();

    /* ── IPC ── */
    if (!ipc_queue.empty()) {
        demo_ipc::QueueClient ipc(ipc_queue);
        ipc.send("=== PageRank (AGILE) Results ===");
        { std::ostringstream os; os << "Graph: " << node_num << " nodes, " << edge_num << " edges"; ipc.send(os.str()); }
        { std::ostringstream os; os << "Iterations: " << warm_itr; ipc.send(os.str()); }
        { std::ostringstream os; os << "COLD kernel time: " << cold_time << " s"; ipc.send(os.str()); }
        { std::ostringstream os; os << "WARM kernel time: " << warm_time << " s"; ipc.send(os.str()); }
        { std::ostringstream os; os << "Speedup (cold/warm): " << (cold_time / warm_time) << "x"; ipc.send(os.str()); }
        ipc.send_done();
    }

    return 0;
}
