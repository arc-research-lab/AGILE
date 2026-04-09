/*
 * BFS demo — GPU-memory only (no SSD).
 *
 * Loads both offsets and neighbors into GPU global memory from binary files,
 * then runs the same level-synchronous BFS kernel.  This serves as a
 * pure-GPU baseline for comparison with AGILE and BAM SSD-backed BFS.
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

#include "demo_ipc.h"
#include <logger.hpp>

#define MYINFINITY 0xFFFFFFFFu

/* ── BFS kernel ─────────────────────────────────────────────── */
__global__
void bfs_kernel_gpu(const uint32_t *neighbors,
                    const uint32_t *offsets,
                    uint32_t *node_levels,
                    uint32_t node_num,
                    uint32_t level,
                    uint32_t *changed) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= node_num) return;

    if (node_levels[tid] == level) {
        uint32_t start = offsets[tid];
        uint32_t end   = offsets[tid + 1];
        for (uint32_t j = start; j < end; j++) {
            uint32_t neighbor = neighbors[j];
            if (node_levels[neighbor] == MYINFINITY) {
                node_levels[neighbor] = level + 1;
                *changed = 1;
            }
        }
    }
}

/* ── Parse graph-gen .info.txt ─────────────────────────────── */
static std::unordered_map<std::string, std::string>
parse_info(const std::string &path) {
    std::unordered_map<std::string, std::string> kv;
    std::ifstream in(path);
    if (!in) {
        LOG_ERROR("BFS-GPU", "Cannot open info file: %s", path.c_str());
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

/* ── Read offsets (supports 4 or 8-byte on disk → uint32) ──── */
static std::vector<uint32_t>
read_offsets_file(const std::string &path, uint64_t count,
                  unsigned int offset_size) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        LOG_ERROR("BFS-GPU", "Cannot open offsets file: %s", path.c_str());
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
        LOG_ERROR("BFS-GPU", "Unsupported offset_size: %u", offset_size);
        std::exit(1);
    }
    return offsets;
}

/* ── Read neighbors binary file ────────────────────────────── */
static std::vector<uint32_t>
read_neighbors_file(const std::string &path, uint64_t count) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        LOG_ERROR("BFS-GPU", "Cannot open neighbors file: %s", path.c_str());
        std::exit(1);
    }
    std::vector<uint32_t> neighbors(count);
    in.read(reinterpret_cast<char *>(neighbors.data()),
            count * sizeof(uint32_t));
    return neighbors;
}


int main(int argc, char **argv) {
    CLI::App app{"GPU-memory BFS (no SSD) Demo"};

    /* IPC */
    std::string ipc_queue;
    app.add_option("--ipc-queue", ipc_queue,
                   "IPC message queue name (set by server)");

    /* Info file */
    std::string info_file;
    app.add_option("-i,--info", info_file,
                   "Path to the .info.txt produced by graph-gen");

    /* Parallelism */
    unsigned int thread_dim = 1024;

    /* Graph config */
    unsigned int node_num   = 0;
    uint64_t     edge_num   = 0;
    unsigned int start_node = 0;
    std::string  offset_file;
    std::string  neighbors_file;
    std::string  output_file = "res-bfs-gpu.bin";

    app.add_option("-t,--threads", thread_dim, "Threads per block");
    app.add_option("--nodes", node_num, "Number of nodes");
    app.add_option("--edges", edge_num, "Number of edges (directed)");
    app.add_option("-s,--start", start_node, "Start node for BFS");
    app.add_option("--offset-file", offset_file, "Offsets binary file");
    app.add_option("--neighbors-file", neighbors_file, "Neighbors binary file");
    app.add_option("-o,--output", output_file, "Output BFS result file");

    CLI11_PARSE(app, argc, argv);

    /* ── Resolve from info file ── */
    unsigned int offset_size = 4;
    if (!info_file.empty()) {
        auto kv = parse_info(info_file);
        auto get = [&](const std::string &key) -> std::string {
            auto it = kv.find(key);
            if (it == kv.end()) {
                LOG_ERROR("BFS-GPU", "Missing key '%s' in %s",
                          key.c_str(), info_file.c_str());
                std::exit(1);
            }
            return it->second;
        };

        auto info_dir = std::filesystem::path(info_file).parent_path();

        if (offset_file.empty()) {
            offset_file = get("offsets_file");
            if (!offset_file.empty() && offset_file[0] != '/' && !info_dir.empty())
                offset_file = (info_dir / offset_file).string();
        }
        if (neighbors_file.empty()) {
            neighbors_file = get("neighbors_file");
            if (!neighbors_file.empty() && neighbors_file[0] != '/' && !info_dir.empty())
                neighbors_file = (info_dir / neighbors_file).string();
        }
        if (node_num == 0) node_num = std::stoul(get("num_nodes"));
        if (edge_num == 0) {
            auto it = kv.find("num_edges_directed");
            edge_num = (it != kv.end()) ? std::stoull(it->second)
                                        : std::stoull(get("num_edges"));
        }
        offset_size = std::stoul(get("offset_size"));
    }

    if (offset_file.empty() || neighbors_file.empty() || node_num == 0) {
        LOG_ERROR("BFS-GPU", "Provide --info or --offset-file, --neighbors-file and --nodes");
        return 1;
    }
    if (start_node >= node_num) {
        LOG_ERROR("BFS-GPU", "start_node %u >= node_num %u", start_node, node_num);
        return 1;
    }

    LOG_INFO("BFS-GPU", "nodes: %u, edges(directed): %lu, start: %u",
             node_num, (unsigned long)edge_num, start_node);
    LOG_INFO("BFS-GPU", "offsets: %s", offset_file.c_str());
    LOG_INFO("BFS-GPU", "neighbors: %s", neighbors_file.c_str());

    /* ── Load graph into host memory ── */
    LOG_INFO("BFS-GPU", "Loading offsets...");
    auto h_offsets = read_offsets_file(offset_file, node_num + 1, offset_size);

    LOG_INFO("BFS-GPU", "Loading neighbors...");
    auto h_neighbors = read_neighbors_file(neighbors_file, edge_num);

    /* ── Initialize host labels ── */
    std::vector<uint32_t> h_labels(node_num, MYINFINITY);
    h_labels[start_node] = 0;

    /* ── Allocate & copy to GPU ── */
    uint32_t *d_offsets, *d_neighbors, *d_labels, *d_changed;
    cudaMalloc(&d_offsets,   (node_num + 1) * sizeof(uint32_t));
    cudaMalloc(&d_neighbors, edge_num * sizeof(uint32_t));
    cudaMalloc(&d_labels,    node_num * sizeof(uint32_t));
    cudaMalloc(&d_changed,   sizeof(uint32_t));

    LOG_INFO("BFS-GPU", "Copying graph to GPU (%.1f MB offsets + %.1f MB neighbors)...",
             (node_num + 1) * sizeof(uint32_t) / 1e6,
             edge_num * sizeof(uint32_t) / 1e6);

    cudaMemcpy(d_offsets, h_offsets.data(),
               (node_num + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighbors, h_neighbors.data(),
               edge_num * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, h_labels.data(),
               node_num * sizeof(uint32_t), cudaMemcpyHostToDevice);

    unsigned int grid_dim = (node_num + thread_dim - 1) / thread_dim;
    LOG_INFO("BFS-GPU", "Starting BFS... grid=%u threads=%u", grid_dim, thread_dim);

    /* ── BFS loop ── */
    double total_itr_time = 0;
    unsigned int level = 0;
    uint32_t changed = 0;

    auto t_start = std::chrono::high_resolution_clock::now();
    do {
        changed = 0;
        cudaMemcpy(d_changed, &changed, sizeof(uint32_t), cudaMemcpyHostToDevice);

        auto s0 = std::chrono::high_resolution_clock::now();
        bfs_kernel_gpu<<<grid_dim, thread_dim>>>(
            d_neighbors, d_offsets, d_labels,
            node_num, level, d_changed);
        cudaDeviceSynchronize();
        auto e0 = std::chrono::high_resolution_clock::now();

        double itr_time =
            std::chrono::duration_cast<std::chrono::duration<double>>(e0 - s0)
                .count();
        total_itr_time += itr_time;

        cudaMemcpy(&changed, d_changed, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        LOG_INFO("BFS-GPU", "level %u  changed=%u  time=%.6f s",
                 level, changed, itr_time);
        level++;
    } while (changed);
    auto t_end = std::chrono::high_resolution_clock::now();

    double total_time =
        std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start)
            .count();

    LOG_INFO("BFS-GPU", "BFS done: %u levels, kernel time=%.6f s, total=%.6f s",
             level, total_itr_time, total_time);

    /* ── Copy results back ── */
    cudaMemcpy(h_labels.data(), d_labels,
               node_num * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    unsigned int visited = 0;
    for (unsigned int i = 0; i < node_num; i++) {
        if (h_labels[i] != MYINFINITY) visited++;
    }
    LOG_INFO("BFS-GPU", "Visited: %u / %u nodes", visited, node_num);

    if (!output_file.empty()) {
        std::ofstream ofs(output_file, std::ios::binary);
        ofs.write(reinterpret_cast<const char *>(h_labels.data()),
                  node_num * sizeof(uint32_t));
        ofs.close();
        LOG_INFO("BFS-GPU", "Result saved to %s", output_file.c_str());
    }

    /* ── Cleanup ── */
    cudaFree(d_offsets);
    cudaFree(d_neighbors);
    cudaFree(d_labels);
    cudaFree(d_changed);

    /* ── IPC ── */
    if (!ipc_queue.empty()) {
        demo_ipc::QueueClient ipc(ipc_queue);

        ipc.send("=== BFS (GPU-memory, no SSD) Results ===");
        {
            std::ostringstream os;
            os << "Graph: " << node_num << " nodes, " << edge_num
               << " edges (directed)";
            ipc.send(os.str());
        }
        {
            std::ostringstream os;
            os << "Start node: " << start_node;
            ipc.send(os.str());
        }
        {
            std::ostringstream os;
            os << "Levels: " << level;
            ipc.send(os.str());
        }
        {
            std::ostringstream os;
            os << "Visited: " << visited << " / " << node_num << " nodes";
            ipc.send(os.str());
        }
        {
            std::ostringstream os;
            os << "Kernel time: " << total_itr_time << " s";
            ipc.send(os.str());
        }
        {
            std::ostringstream os;
            os << "Total time: " << total_time << " s";
            ipc.send(os.str());
        }

        ipc.send_done();
    }

    return 0;
}
