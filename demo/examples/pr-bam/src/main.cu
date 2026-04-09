/*
 * BAM PageRank demo — iterative SpMV on SSD-backed CSR graph.
 *
 * SSD layout: [neighbors (col indices)] [normalised weights]
 *             ^offset 0                 ^weight_offset
 *
 * Cold/warm comparison like the AGILE version.
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

/* ── PageRank SpMV kernel (BAM) ────────────────────────────── */
__global__
void pagerank_kernel_bam(array_d_t<uint32_t> *da,
                         unsigned int *offsets,
                         unsigned int nodes,
                         unsigned int weight_offset,
                         float *vec,
                         float *output_vec,
                         float damping,
                         float *norm2) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= nodes) return;

    output_vec[tid] = 0;
    unsigned int start = offsets[tid];
    unsigned int end   = offsets[tid + 1];
    for (unsigned int j = start; j < end; ++j) {
        unsigned int col = da->seq_read(j);
        unsigned int val = da->seq_read(weight_offset + j);
        output_vec[tid] += __uint_as_float(val) * vec[col];
    }
    output_vec[tid] *= damping;
    output_vec[tid] += (1.0f - damping) / nodes;
    float diff = output_vec[tid] - vec[tid];
    atomicAdd(norm2, diff * diff);
}

/* ── Parse info.txt ────────────────────────────────────────── */
static std::unordered_map<std::string, std::string>
parse_info(const std::string &path) {
    std::unordered_map<std::string, std::string> kv;
    std::ifstream in(path);
    if (!in) { LOG_ERROR("PR-BAM", "Cannot open info file: %s", path.c_str()); std::exit(1); }
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

/* ── Read offsets ──────────────────────────────────────────── */
static std::vector<uint32_t>
read_offsets_file(const std::string &path, uint64_t count, unsigned int offset_size) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { LOG_ERROR("PR-BAM", "Cannot open offsets: %s", path.c_str()); std::exit(1); }
    std::vector<uint32_t> offsets(count);
    if (offset_size == 4) {
        in.read(reinterpret_cast<char *>(offsets.data()), count * 4);
    } else if (offset_size == 8) {
        std::vector<uint64_t> tmp(count);
        in.read(reinterpret_cast<char *>(tmp.data()), count * 8);
        for (uint64_t i = 0; i < count; ++i) offsets[i] = static_cast<uint32_t>(tmp[i]);
    }
    return offsets;
}


int main(int argc, char **argv) {
    CLI::App app{"BAM PageRank Demo"};

    std::string ipc_queue;
    app.add_option("--ipc-queue", ipc_queue, "IPC message queue name");

    std::string info_file;
    app.add_option("-i,--info", info_file, "Path to .info.txt");

    /* BAM config */
    std::string nvme_device = "/dev/libnvm0";
    unsigned int page_size  = 4096;
    uint64_t     num_pages  = 65536 * 8;
    unsigned int queue_num  = 15;
    unsigned int queue_depth = 512;
    unsigned int gpu_id     = 0;
    unsigned int ns_id      = 1;

    /* Parallelism */
    unsigned int thread_dim = 64;

    /* Graph config */
    unsigned int node_num = 0;
    uint64_t     edge_num = 0;
    unsigned int weight_offset = 0;
    std::string  offset_file;
    std::string  output_file = "res-pr-bam.bin";

    /* PageRank config */
    float damping = 0.85f;
    unsigned int max_itr = 20;
    float error_thresh = 1e-6f;

    app.add_option("-d,--dev", nvme_device, "BAM NVMe device");
    app.add_option("--page-size", page_size, "Page size");
    app.add_option("--pages", num_pages, "Number of cache pages");
    app.add_option("-q,--queues", queue_num, "NVMe queues");
    app.add_option("--queue-depth", queue_depth, "Queue depth");
    app.add_option("-g,--gpu", gpu_id, "CUDA device ID");
    app.add_option("--ns", ns_id, "NVMe namespace ID");
    app.add_option("-t,--threads", thread_dim, "Threads per block");
    app.add_option("--nodes", node_num, "Number of nodes");
    app.add_option("--edges", edge_num, "Number of edges (directed)");
    app.add_option("--weight-offset", weight_offset, "Weight offset (uint32 elements)");
    app.add_option("--offset-file", offset_file, "Offsets binary file");
    app.add_option("-o,--output", output_file, "Output file");
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
            if (it == kv.end()) { LOG_ERROR("PR-BAM", "Missing key '%s'", key.c_str()); std::exit(1); }
            return it->second;
        };
        auto info_dir = std::filesystem::path(info_file).parent_path();
        if (offset_file.empty()) {
            offset_file = get("offsets_file");
            if (!offset_file.empty() && offset_file[0] != '/' && !info_dir.empty())
                offset_file = (info_dir / offset_file).string();
        }
        if (node_num == 0) node_num = std::stoul(get("num_nodes"));
        if (edge_num == 0) {
            auto it = kv.find("num_edges_directed");
            edge_num = (it != kv.end()) ? std::stoull(it->second) : std::stoull(get("num_edges"));
        }
        offset_size = std::stoul(get("offset_size"));
        if (weight_offset == 0) {
            auto it = kv.find("neighbors_filesize");
            if (it != kv.end())
                weight_offset = static_cast<unsigned int>(std::stoull(it->second) / sizeof(uint32_t));
        }
    }

    if (offset_file.empty() || node_num == 0) {
        LOG_ERROR("PR-BAM", "Provide --info or manual options"); return 1;
    }

    LOG_INFO("PR-BAM", "nodes: %u, edges: %lu, weight_offset: %u",
             node_num, (unsigned long)edge_num, weight_offset);
    LOG_INFO("PR-BAM", "device: %s, pages: %lu, queues: %u",
             nvme_device.c_str(), (unsigned long)num_pages, queue_num);

    /* ── Load offsets ── */
    LOG_INFO("PR-BAM", "Loading offsets...");
    auto h_offsets_vec = read_offsets_file(offset_file, node_num + 1, offset_size);

    /* ── Setup BAM ── */
    cuda_err_chk(cudaSetDevice(gpu_id));
    Controller ctrl(nvme_device.c_str(), ns_id, gpu_id, queue_depth, queue_num);
    std::vector<Controller *> ctrls_vec = {&ctrl};
    page_cache_t h_pc(page_size, num_pages, gpu_id, ctrl, 64, ctrls_vec);

    /* Range covers neighbors + weights */
    uint64_t total_elems = weight_offset + edge_num; /* neighbors + weights */
    uint64_t t_size = total_elems * sizeof(uint32_t);
    uint64_t total_pages = (t_size / page_size) + 1;
    range_t<uint32_t> h_range(0, total_elems, 0, total_pages, 0, page_size, &h_pc, gpu_id);
    std::vector<range_t<uint32_t> *> vr = {&h_range};
    array_t<uint32_t> nvme_mem(total_elems, 0, vr, gpu_id);

    LOG_INFO("PR-BAM", "BAM initialized (total_elems=%lu)", (unsigned long)total_elems);

    /* ── Allocate GPU arrays ── */
    uint32_t *d_offsets;
    float *d_vec, *d_out_vec, *d_norm2;
    cuda_err_chk(cudaMalloc(&d_offsets, (node_num + 1) * sizeof(uint32_t)));
    cuda_err_chk(cudaMalloc(&d_vec, node_num * sizeof(float)));
    cuda_err_chk(cudaMalloc(&d_out_vec, node_num * sizeof(float)));
    cuda_err_chk(cudaMalloc(&d_norm2, sizeof(float)));
    cuda_err_chk(cudaMemcpy(d_offsets, h_offsets_vec.data(),
                            (node_num + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));

    float initial_val = 1.0f / node_num;
    std::vector<float> h_vec(node_num, initial_val);

    unsigned int grid_dim = node_num / thread_dim + 1;

    /* ── Helper: run PR iterations ── */
    auto run_pr = [&](const char *tag) -> std::pair<double, unsigned int> {
        cuda_err_chk(cudaMemcpy(d_vec, h_vec.data(), node_num * sizeof(float), cudaMemcpyHostToDevice));
        double total_kernel_time = 0;
        unsigned int itr = 0;
        float h_norm2 = 0;
        do {
            h_norm2 = 0;
            cuda_err_chk(cudaMemcpy(d_norm2, &h_norm2, sizeof(float), cudaMemcpyHostToDevice));
            auto s0 = std::chrono::high_resolution_clock::now();
            pagerank_kernel_bam<<<grid_dim, thread_dim>>>(
                nvme_mem.d_array_ptr, d_offsets, node_num, weight_offset,
                d_vec, d_out_vec, damping, d_norm2);
            cuda_err_chk(cudaGetLastError());
            cuda_err_chk(cudaDeviceSynchronize());
            auto e0 = std::chrono::high_resolution_clock::now();
            cuda_err_chk(cudaMemcpy(d_vec, d_out_vec, node_num * sizeof(float), cudaMemcpyDeviceToDevice));
            cuda_err_chk(cudaMemcpy(&h_norm2, d_norm2, sizeof(float), cudaMemcpyDeviceToHost));
            h_norm2 = sqrtf(h_norm2);
            double itr_time = std::chrono::duration_cast<std::chrono::duration<double>>(e0 - s0).count();
            total_kernel_time += itr_time;
            LOG_INFO("PR-BAM", "[%s] itr %u  norm=%.6e  time=%.6f s", tag, itr, h_norm2, itr_time);
            itr++;
        } while (itr < max_itr && h_norm2 > error_thresh);
        return {total_kernel_time, itr};
    };

    /* ── COLD ── */
    LOG_INFO("PR-BAM", "=== Pass 1: COLD (loading from SSD) ===");
    auto [cold_time, cold_itr] = run_pr("COLD");

    /* ── WARM ── */
    LOG_INFO("PR-BAM", "=== Pass 2: WARM (data in page cache) ===");
    auto [warm_time, warm_itr] = run_pr("WARM");

    LOG_INFO("PR-BAM", "=== Summary ===");
    LOG_INFO("PR-BAM", "  COLD kernel: %.6f s   WARM kernel: %.6f s   Speedup: %.2fx",
             cold_time, warm_time, cold_time / warm_time);

    /* ── Save ── */
    std::vector<float> h_result(node_num);
    cuda_err_chk(cudaMemcpy(h_result.data(), d_vec, node_num * sizeof(float), cudaMemcpyDeviceToHost));

    if (!output_file.empty()) {
        std::ofstream ofs(output_file, std::ios::binary);
        ofs.write(reinterpret_cast<const char *>(h_result.data()), node_num * sizeof(float));
        ofs.close();
        LOG_INFO("PR-BAM", "Result saved to %s", output_file.c_str());
    }

    /* ── Cleanup ── */
    cuda_err_chk(cudaFree(d_offsets));
    cuda_err_chk(cudaFree(d_vec));
    cuda_err_chk(cudaFree(d_out_vec));
    cuda_err_chk(cudaFree(d_norm2));

    /* ── IPC ── */
    if (!ipc_queue.empty()) {
        demo_ipc::QueueClient ipc(ipc_queue);
        ipc.send("=== PageRank (BAM) Results ===");
        { std::ostringstream os; os << "Graph: " << node_num << " nodes, " << edge_num << " edges"; ipc.send(os.str()); }
        { std::ostringstream os; os << "Iterations: " << warm_itr; ipc.send(os.str()); }
        { std::ostringstream os; os << "COLD kernel time: " << cold_time << " s"; ipc.send(os.str()); }
        { std::ostringstream os; os << "WARM kernel time: " << warm_time << " s"; ipc.send(os.str()); }
        { std::ostringstream os; os << "Speedup (cold/warm): " << (cold_time / warm_time) << "x"; ipc.send(os.str()); }
        ipc.send_done();
    }

    return 0;
}
