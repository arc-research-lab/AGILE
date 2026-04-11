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


__global__ void bfs_kernel(AGILE_CTRL *ctrl,
                           unsigned int node_num, unsigned int level,
                           unsigned int *changed, unsigned int *offsets,
                           unsigned int *node_levels,
                           unsigned int thread_num) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    AgileLockChain chain;

    if (tid >= node_num) return;

    if (node_levels[tid] == level) {
        auto agileArr = ctrl->getArrayWrap<unsigned int>(chain);
        for (unsigned int j = offsets[tid]; j < offsets[tid + 1]; j++) {
            unsigned int neighbor = agileArr[0][j];
            // if (level == 0) {
            //     printf("tid=%u j=%u neighbor=%u\n", tid, j, neighbor);
            // }
            if (node_levels[neighbor] == (unsigned int)-1) {
                node_levels[neighbor] = level + 1;
                *changed = 1;
            }
        }
    }
}


// ---------------------------------------------------------------------------
// Parse a graph-gen .info.txt file into key-value pairs.
// ---------------------------------------------------------------------------
static std::unordered_map<std::string, std::string>
parse_info(const std::string &path) {
    std::unordered_map<std::string, std::string> kv;
    std::ifstream in(path);
    if (!in) {
        LOG_ERROR("BFS", "Cannot open info file: %s", path.c_str());
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

// ---------------------------------------------------------------------------
// Read offsets from binary file.  offset_size may be 4 or 8.
// The GPU kernel uses unsigned int offsets, so we always produce uint32.
// ---------------------------------------------------------------------------
static std::vector<unsigned int>
read_offsets_file(const std::string &path, uint64_t count,
                  unsigned int offset_size) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        LOG_ERROR("BFS", "Cannot open offsets file: %s", path.c_str());
        std::exit(1);
    }
    std::vector<unsigned int> offsets(count);
    if (offset_size == 4) {
        in.read(reinterpret_cast<char *>(offsets.data()),
                count * sizeof(uint32_t));
    } else if (offset_size == 8) {
        std::vector<uint64_t> tmp(count);
        in.read(reinterpret_cast<char *>(tmp.data()),
                count * sizeof(uint64_t));
        for (uint64_t i = 0; i < count; ++i)
            offsets[i] = static_cast<unsigned int>(tmp[i]);
    } else {
        LOG_ERROR("BFS", "Unsupported offset_size: %u", offset_size);
        std::exit(1);
    }
    return offsets;
}


int main(int argc, char **argv) {
    CLI::App app{"AGILE BFS (Breadth-First Search) Demo"};

    // IPC
    std::string ipc_queue;
    app.add_option("--ipc-queue", ipc_queue,
                   "IPC message queue name (set by server)");

    // Info file (auto-resolves graph params)
    std::string info_file;
    app.add_option("-i,--info", info_file,
                   "Path to the .info.txt produced by graph-gen");

    // Agile config
    unsigned int slot_size = 4096;
    unsigned int gpu_slot_num = 65536 * 8;

    // NVMe config
    std::string nvme_device = "/dev/AGILE-NVMe-0000:01:00.0";
    unsigned int queue_num = 15;
    unsigned int queue_depth = 512;
    unsigned int ssd_blk_offset = 0;

    // Parallelism config
    unsigned int block_dim = 32;
    unsigned int thread_dim = 1024;
    unsigned int agile_dim = 1;

    // Graph config (overridable, or auto-filled from info file)
    unsigned int node_num = 0;
    unsigned int edge_num = 0;
    unsigned int ssd_block_num = 1048576;
    unsigned int start_node = 0;
    std::string offset_file;
    std::string output_file = "res-bfs.bin";

    app.add_option("-d,--dev", nvme_device, "NVMe device path");
    app.add_option("--slot-size", slot_size, "Slot size of the cache");
    app.add_option("--gpu-slots", gpu_slot_num, "Number of GPU cache slots");
    app.add_option("-q,--queues", queue_num, "Number of NVMe queue pairs");
    app.add_option("--queue-depth", queue_depth, "Depth of each NVMe queue");
    app.add_option("--blk-offset", ssd_blk_offset, "SSD block offset");
    app.add_option("-b,--blocks", block_dim, "Block dimension");
    app.add_option("-t,--threads", thread_dim, "Thread dimension");
    app.add_option("--agile-dim", agile_dim, "Agile service blocks");
    app.add_option("--nodes", node_num, "Number of nodes");
    app.add_option("--edges", edge_num, "Number of edges (directed)");
    app.add_option("--ssd-blocks", ssd_block_num, "Number of SSD blocks");
    app.add_option("-s,--start", start_node, "Start node for BFS");
    app.add_option("--offset-file", offset_file, "Offsets binary file");
    app.add_option("-o,--output", output_file, "Output BFS result file");

    CLI11_PARSE(app, argc, argv);

    // ── Resolve from info file if provided ──
    unsigned int offset_size = 4;  // default: 4 bytes
    if (!info_file.empty()) {
        auto kv = parse_info(info_file);
        auto get = [&](const std::string &key) -> std::string {
            auto it = kv.find(key);
            if (it == kv.end()) {
                LOG_ERROR("BFS", "Missing key '%s' in %s", key.c_str(),
                          info_file.c_str());
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
            // CSR uses directed edge count
            auto it = kv.find("num_edges_directed");
            edge_num = (it != kv.end()) ? std::stoul(it->second)
                                        : std::stoul(get("num_edges"));
        }
        offset_size = std::stoul(get("offset_size"));

        // ssd_block_num: neighbors file size / slot_size
        auto it = kv.find("neighbors_filesize");
        if (it != kv.end()) {
            uint64_t neighbors_filesize = std::stoull(it->second);
            ssd_block_num = static_cast<unsigned int>(neighbors_filesize / slot_size);
            LOG_INFO("BFS", "Auto ssd_block_num from filesize: %u (filesize=%lu, slot=%u)",
                     ssd_block_num, neighbors_filesize, slot_size);
        }
    }

    if (offset_file.empty() || node_num == 0) {
        LOG_ERROR("BFS", "Provide --info or --offset-file and --nodes");
        return 1;
    }
    if (start_node >= node_num) {
        LOG_ERROR("BFS", "start_node %u >= node_num %u", start_node, node_num);
        return 1;
    }

    LOG_INFO("BFS", "nodes: %u, edges(directed): %u, start: %u",
             node_num, edge_num, start_node);
    LOG_INFO("BFS", "offset_file: %s (offset_size=%u)", offset_file.c_str(),
             offset_size);
    LOG_INFO("BFS", "device: %s, queues: %u, slot_size: %u, gpu_slots: %u",
             nvme_device.c_str(), queue_num, slot_size, gpu_slot_num);

    // ── Load offsets from file ──
    LOG_INFO("BFS", "Loading offsets...");
    auto h_offsets_vec =
        read_offsets_file(offset_file, node_num + 1, offset_size);
    unsigned int *h_offsets = h_offsets_vec.data();

    // ── Setup AGILE (must precede cudaMalloc to match expected init order) ──
    AGILE_HOST host(0, slot_size);

    CPU_CACHE_IMPL c_cache(0, slot_size);
    SHARE_TABLE_IMPL w_table(gpu_slot_num / 4);
    GPU_CACHE_IMPL g_cache(gpu_slot_num, slot_size, ssd_block_num);

    host.setGPUCache(g_cache);
    host.setCPUCache(c_cache);
    host.setShareTable(w_table);

    host.addNvmeDev(nvme_device, ssd_blk_offset, queue_num, queue_depth);
    host.initNvme();

    // ── Allocate host/device arrays ──
    unsigned int *h_node_levels =
        (unsigned int *)malloc(node_num * sizeof(unsigned int));
    for (unsigned int i = 0; i < node_num; i++)
        h_node_levels[i] = (unsigned int)-1;
    h_node_levels[start_node] = 0;

    unsigned int *d_changed, *d_offsets, *d_node_levels;
    cuda_err_chk(cudaMalloc(&d_changed, sizeof(unsigned int)));
    cuda_err_chk(cudaMalloc(&d_offsets, (node_num + 1) * sizeof(unsigned int)));
    cuda_err_chk(cudaMalloc(&d_node_levels, node_num * sizeof(unsigned int)));
    cuda_err_chk(cudaMemcpy(d_offsets, h_offsets,
                            (node_num + 1) * sizeof(unsigned int),
                            cudaMemcpyHostToDevice));
    cuda_err_chk(cudaMemcpy(d_node_levels, h_node_levels,
                            node_num * sizeof(unsigned int),
                            cudaMemcpyHostToDevice));

    unsigned int blockDim_val = node_num / thread_dim + 1;
    host.configParallelism(blockDim_val, thread_dim, agile_dim);
    host.queryOccupancy(bfs_kernel, 1024, 0);

    host.initializeAgile();
    auto *ctrl = host.getAgileCtrlDevicePtr();

    // ── Run BFS ──
    LOG_INFO("BFS", "Starting BFS... grid=%u threads=%u", blockDim_val, thread_dim);
    host.startAgile();

    std::chrono::high_resolution_clock::time_point t_start, t_end, s0, e0;

    // ────────────────────────────────────────────────────────────
    // Pass 1: COLD — data loaded from SSD into software cache
    // ────────────────────────────────────────────────────────────
    double cold_itr_time = 0;
    unsigned int cold_level = 0;
    unsigned int changed = 0;

    LOG_INFO("BFS", "=== Pass 1: COLD (loading from SSD) ===");
    t_start = std::chrono::high_resolution_clock::now();
    do {
        changed = 0;
        cuda_err_chk(cudaMemcpy(d_changed, &changed, sizeof(unsigned int),
                                cudaMemcpyHostToDevice));
        s0 = std::chrono::high_resolution_clock::now();
        host.runKernel(bfs_kernel, ctrl, node_num, cold_level, d_changed, d_offsets,
                       d_node_levels, block_dim * thread_dim);
        e0 = std::chrono::high_resolution_clock::now();
        double itr_time =
            std::chrono::duration_cast<std::chrono::duration<double>>(e0 - s0)
                .count();
        cold_itr_time += itr_time;

        cuda_err_chk(cudaMemcpy(&changed, d_changed, sizeof(unsigned int),
                                cudaMemcpyDeviceToHost));
        LOG_INFO("BFS", "[COLD] level %u  changed=%u  time=%.6f s", cold_level, changed,
                 itr_time);
        cold_level++;
    } while (changed);
    t_end = std::chrono::high_resolution_clock::now();
    double cold_total_time =
        std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start)
            .count();
    LOG_INFO("BFS", "COLD done: %u levels, kernel time=%.6f s, total=%.6f s",
             cold_level, cold_itr_time, cold_total_time);

    // ────────────────────────────────────────────────────────────
    // Pass 2: WARM — data already in software cache
    // Reset node_levels and re-run BFS
    // ────────────────────────────────────────────────────────────
    cuda_err_chk(cudaMemcpy(d_node_levels, h_node_levels,
                            node_num * sizeof(unsigned int),
                            cudaMemcpyHostToDevice));

    double warm_itr_time = 0;
    unsigned int warm_level = 0;

    LOG_INFO("BFS", "=== Pass 2: WARM (data in software cache) ===");
    t_start = std::chrono::high_resolution_clock::now();
    do {
        changed = 0;
        cuda_err_chk(cudaMemcpy(d_changed, &changed, sizeof(unsigned int),
                                cudaMemcpyHostToDevice));
        s0 = std::chrono::high_resolution_clock::now();
        host.runKernel(bfs_kernel, ctrl, node_num, warm_level, d_changed, d_offsets,
                       d_node_levels, block_dim * thread_dim);
        e0 = std::chrono::high_resolution_clock::now();
        double itr_time =
            std::chrono::duration_cast<std::chrono::duration<double>>(e0 - s0)
                .count();
        warm_itr_time += itr_time;

        cuda_err_chk(cudaMemcpy(&changed, d_changed, sizeof(unsigned int),
                                cudaMemcpyDeviceToHost));
        LOG_INFO("BFS", "[WARM] level %u  changed=%u  time=%.6f s", warm_level, changed,
                 itr_time);
        warm_level++;
    } while (changed);
    t_end = std::chrono::high_resolution_clock::now();
    double warm_total_time =
        std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start)
            .count();

    host.stopAgile();

    unsigned int level = warm_level;
    LOG_INFO("BFS", "WARM done: %u levels, kernel time=%.6f s, total=%.6f s",
             level, warm_itr_time, warm_total_time);
    LOG_INFO("BFS", "=== Summary ===");
    LOG_INFO("BFS", "  COLD kernel: %.6f s   WARM kernel: %.6f s   Speedup: %.2fx",
             cold_itr_time, warm_itr_time, cold_itr_time / warm_itr_time);

    double total_itr_time = warm_itr_time;
    double total_time = warm_total_time;

    // ── Copy results back and save ──
    cuda_err_chk(cudaMemcpy(h_node_levels, d_node_levels,
                            node_num * sizeof(unsigned int),
                            cudaMemcpyDeviceToHost));

    // Count visited nodes
    unsigned int visited = 0;
    for (unsigned int i = 0; i < node_num; i++) {
        if (h_node_levels[i] != (unsigned int)-1) visited++;
    }
    LOG_INFO("BFS", "Visited: %u / %u nodes", visited, node_num);

    if (!output_file.empty()) {
        std::ofstream ofs(output_file, std::ios::binary);
        ofs.write(reinterpret_cast<const char *>(h_node_levels),
                  node_num * sizeof(unsigned int));
        ofs.close();
        LOG_INFO("BFS", "Result saved to %s", output_file.c_str());
    }

    // ── Cleanup ──
    cuda_err_chk(cudaFree(d_changed));
    cuda_err_chk(cudaFree(d_offsets));
    cuda_err_chk(cudaFree(d_node_levels));
    free(h_node_levels);
    host.closeNvme();

    // ── Send results via IPC (if launched by the server) ──
    if (!ipc_queue.empty()) {
        demo_ipc::QueueClient ipc(ipc_queue);

        ipc.send("=== BFS (AGILE) Results ===");
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
            os << "COLD kernel time: " << cold_itr_time << " s";
            ipc.send(os.str());
        }
        {
            std::ostringstream os;
            os << "WARM kernel time: " << warm_itr_time << " s";
            ipc.send(os.str());
        }
        {
            std::ostringstream os;
            os << "Speedup (cold/warm): " << (cold_itr_time / warm_itr_time) << "x";
            ipc.send(os.str());
        }

        ipc.send_done();
    }

    return 0;
}
