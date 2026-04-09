/*
 * GPU-Memory-Only PageRank demo — iterative SpMV with all data in GPU RAM.
 *
 * Loads CSR offsets, neighbors and pre-computed column-normalised weights
 * from host files, copies to GPU, and iterates.
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
#include <cuda_runtime.h>

#include "demo_ipc.h"
#include <logger.hpp>

#define cuda_err_chk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
static inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error %s @ %s:%d\n", cudaGetErrorString(code), file, line);
        std::exit(1);
    }
}

/* ── PageRank SpMV kernel (GPU memory) ─────────────────────── */
__global__
void pagerank_kernel_gpu(unsigned int *neighbors,
                         float *weights,
                         unsigned int *offsets,
                         unsigned int nodes,
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
        unsigned int col = neighbors[j];
        float w = weights[j];
        output_vec[tid] += w * vec[col];
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
    if (!in) { LOG_ERROR("PR-GPU", "Cannot open info file: %s", path.c_str()); std::exit(1); }
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

/* ── Read binary file into host vector ────────────────────── */
template<typename T>
static std::vector<T> read_binary(const std::string &path, uint64_t count) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { LOG_ERROR("PR-GPU", "Cannot open %s", path.c_str()); std::exit(1); }
    std::vector<T> v(count);
    in.read(reinterpret_cast<char *>(v.data()), count * sizeof(T));
    return v;
}

static std::vector<uint32_t>
read_offsets_file(const std::string &path, uint64_t count, unsigned int offset_size) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { LOG_ERROR("PR-GPU", "Cannot open offsets: %s", path.c_str()); std::exit(1); }
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
    CLI::App app{"GPU-Memory PageRank Demo"};

    std::string ipc_queue;
    app.add_option("--ipc-queue", ipc_queue, "IPC message queue name");

    std::string info_file;
    app.add_option("-i,--info", info_file, "Path to .info.txt");

    unsigned int gpu_id    = 0;
    unsigned int thread_dim = 1024;

    unsigned int node_num  = 0;
    uint64_t     edge_num  = 0;
    std::string  offset_file;
    std::string  neighbor_file;
    std::string  weight_file;
    std::string  output_file = "res-pr-gpu.bin";

    float damping = 0.85f;
    unsigned int max_itr = 20;
    float error_thresh = 1e-6f;

    app.add_option("-g,--gpu", gpu_id, "CUDA device ID");
    app.add_option("-t,--threads", thread_dim, "Threads per block");
    app.add_option("--nodes", node_num, "Number of nodes");
    app.add_option("--edges", edge_num, "Number of edges (directed)");
    app.add_option("--offset-file", offset_file, "Offsets binary file");
    app.add_option("--neighbor-file", neighbor_file, "Neighbors binary file");
    app.add_option("--weight-file", weight_file, "Normalised weights binary file");
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
            if (it == kv.end()) { LOG_ERROR("PR-GPU", "Missing key '%s'", key.c_str()); std::exit(1); }
            return it->second;
        };
        auto info_dir = std::filesystem::path(info_file).parent_path();
        auto resolve = [&](const std::string &p) -> std::string {
            if (!p.empty() && p[0] != '/' && !info_dir.empty())
                return (info_dir / p).string();
            return p;
        };
        if (offset_file.empty())   offset_file   = resolve(get("offsets_file"));
        if (neighbor_file.empty()) neighbor_file  = resolve(get("neighbors_file"));
        if (weight_file.empty()) {
            auto it = kv.find("weights_file");
            if (it != kv.end()) weight_file = resolve(it->second);
        }
        if (node_num == 0) node_num = std::stoul(get("num_nodes"));
        if (edge_num == 0) {
            auto it = kv.find("num_edges_directed");
            edge_num = (it != kv.end()) ? std::stoull(it->second) : std::stoull(get("num_edges"));
        }
        offset_size = std::stoul(get("offset_size"));
    }

    if (offset_file.empty() || neighbor_file.empty() || node_num == 0) {
        LOG_ERROR("PR-GPU", "Provide --info or manual options"); return 1;
    }
    if (weight_file.empty()) {
        LOG_ERROR("PR-GPU", "Normalised weights file required (--weight-file or weights_file in info)");
        return 1;
    }

    LOG_INFO("PR-GPU", "nodes: %u, edges: %lu", node_num, (unsigned long)edge_num);

    /* ── Load data ── */
    LOG_INFO("PR-GPU", "Loading offsets...");
    auto h_offsets = read_offsets_file(offset_file, node_num + 1, offset_size);

    LOG_INFO("PR-GPU", "Loading neighbors (%lu edges)...", (unsigned long)edge_num);
    auto h_neighbors = read_binary<uint32_t>(neighbor_file, edge_num);

    LOG_INFO("PR-GPU", "Loading weights...");
    auto h_weights = read_binary<float>(weight_file, edge_num);

    /* ── Allocate GPU ── */
    cuda_err_chk(cudaSetDevice(gpu_id));

    uint32_t *d_offsets, *d_neighbors;
    float *d_weights, *d_vec, *d_out_vec, *d_norm2;
    cuda_err_chk(cudaMalloc(&d_offsets,   (node_num + 1) * sizeof(uint32_t)));
    cuda_err_chk(cudaMalloc(&d_neighbors, edge_num * sizeof(uint32_t)));
    cuda_err_chk(cudaMalloc(&d_weights,   edge_num * sizeof(float)));
    cuda_err_chk(cudaMalloc(&d_vec,       node_num * sizeof(float)));
    cuda_err_chk(cudaMalloc(&d_out_vec,   node_num * sizeof(float)));
    cuda_err_chk(cudaMalloc(&d_norm2,     sizeof(float)));

    cuda_err_chk(cudaMemcpy(d_offsets,   h_offsets.data(),
                            (node_num + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    cuda_err_chk(cudaMemcpy(d_neighbors, h_neighbors.data(),
                            edge_num * sizeof(uint32_t), cudaMemcpyHostToDevice));
    cuda_err_chk(cudaMemcpy(d_weights,   h_weights.data(),
                            edge_num * sizeof(float), cudaMemcpyHostToDevice));

    float initial_val = 1.0f / node_num;
    std::vector<float> h_vec(node_num, initial_val);

    unsigned int grid_dim = node_num / thread_dim + 1;

    /* ── Run PageRank ── */
    cuda_err_chk(cudaMemcpy(d_vec, h_vec.data(), node_num * sizeof(float), cudaMemcpyHostToDevice));

    double total_kernel_time = 0;
    unsigned int itr = 0;
    float h_norm2 = 0;
    do {
        h_norm2 = 0;
        cuda_err_chk(cudaMemcpy(d_norm2, &h_norm2, sizeof(float), cudaMemcpyHostToDevice));
        auto s0 = std::chrono::high_resolution_clock::now();
        pagerank_kernel_gpu<<<grid_dim, thread_dim>>>(
            d_neighbors, d_weights, d_offsets, node_num,
            d_vec, d_out_vec, damping, d_norm2);
        cuda_err_chk(cudaDeviceSynchronize());
        auto e0 = std::chrono::high_resolution_clock::now();
        cuda_err_chk(cudaMemcpy(d_vec, d_out_vec, node_num * sizeof(float), cudaMemcpyDeviceToDevice));
        cuda_err_chk(cudaMemcpy(&h_norm2, d_norm2, sizeof(float), cudaMemcpyDeviceToHost));
        h_norm2 = sqrtf(h_norm2);
        double itr_time = std::chrono::duration_cast<std::chrono::duration<double>>(e0 - s0).count();
        total_kernel_time += itr_time;
        LOG_INFO("PR-GPU", "itr %u  norm=%.6e  time=%.6f s", itr, h_norm2, itr_time);
        itr++;
    } while (itr < max_itr && h_norm2 > error_thresh);

    LOG_INFO("PR-GPU", "=== Summary ===");
    LOG_INFO("PR-GPU", "  Iterations: %u   Total kernel time: %.6f s", itr, total_kernel_time);

    /* ── Save ── */
    std::vector<float> h_result(node_num);
    cuda_err_chk(cudaMemcpy(h_result.data(), d_vec, node_num * sizeof(float), cudaMemcpyDeviceToHost));

    if (!output_file.empty()) {
        std::ofstream ofs(output_file, std::ios::binary);
        ofs.write(reinterpret_cast<const char *>(h_result.data()), node_num * sizeof(float));
        ofs.close();
        LOG_INFO("PR-GPU", "Result saved to %s", output_file.c_str());
    }

    /* ── Cleanup ── */
    cuda_err_chk(cudaFree(d_offsets));
    cuda_err_chk(cudaFree(d_neighbors));
    cuda_err_chk(cudaFree(d_weights));
    cuda_err_chk(cudaFree(d_vec));
    cuda_err_chk(cudaFree(d_out_vec));
    cuda_err_chk(cudaFree(d_norm2));

    /* ── IPC ── */
    if (!ipc_queue.empty()) {
        demo_ipc::QueueClient ipc(ipc_queue);
        ipc.send("=== PageRank (GPU Memory) Results ===");
        { std::ostringstream os; os << "Graph: " << node_num << " nodes, " << edge_num << " edges"; ipc.send(os.str()); }
        { std::ostringstream os; os << "Iterations: " << itr; ipc.send(os.str()); }
        { std::ostringstream os; os << "Total kernel time: " << total_kernel_time << " s"; ipc.send(os.str()); }
        ipc.send_done();
    }

    return 0;
}
