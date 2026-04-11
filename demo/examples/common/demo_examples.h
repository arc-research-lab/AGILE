#pragma once
/*
 * demo_examples.h  –  Registry of known demo examples.
 *
 * To add a new example:
 *   1. Build it so the binary lands in the usual CMake output directory.
 *   2. Add an entry to the DEMO_EXAMPLES map below.
 *   3. (Optional) The example should accept --ipc-queue <name> to send
 *      output back to the server through demo_ipc.
 */

#include <string>
#include <unordered_map>
#include <vector>

struct DemoExample {
    std::string id;            // unique slug used in JSON messages
    std::string display_name;  // human-readable label
    std::string binary;        // path relative to the build directory
    /* Extra default CLI args appended when launching.
       The server always adds: --ipc-queue <queue_name>  */
    std::vector<std::string> default_args;
};

/*
 * Central registry.  The key must match the "example" field that the
 * browser sends in JSON, e.g.  {"action":"run","example":"hello-world"}.
 */
inline const std::unordered_map<std::string, DemoExample>& get_demo_examples() {
    static const std::unordered_map<std::string, DemoExample> examples = {
        {"hello-world", {
            "hello-world",
            "Hello World",
            "examples/hello-world/agile_demo_hello_world",
            {} // uses its own defaults
        }},
        {"ctc", {
            "ctc",
            "Compute-Transfer Concurrency",
            "examples/ctc/agile_demo_ctc",
            {} // uses its own defaults
        }},
        {"bfs-agile", {
            "bfs-agile",
            "BFS (AGILE)",
            "examples/bfs/agile_demo_bfs",
            {"-i", "test/bfs_verify.info.txt", "-s", "0"}
        }},
        {"bfs-bam", {
            "bfs-bam",
            "BFS (BAM)",
            "examples/bfs-bam/agile_demo_bfs_bam",
            {"-i", "test/bfs_verify.info.txt", "-s", "0", "--pages", "524288"}
        }},
        {"bfs-gpu", {
            "bfs-gpu",
            "BFS (GPU-memory)",
            "examples/bfs-gpu/agile_demo_bfs_gpu",
            {"-i", "test/bfs_verify.info.txt", "-s", "0"}
        }},
        {"pr-agile", {
            "pr-agile",
            "PageRank (AGILE)",
            "examples/pr-agile/agile_demo_pr",
            {"-i", "test/1gb.info.txt"}
        }},
        {"pr-bam", {
            "pr-bam",
            "PageRank (BAM)",
            "examples/pr-bam/agile_demo_pr_bam",
            {"-i", "test/1gb.info.txt", "--pages", "524288"}
        }},
        {"pr-gpu", {
            "pr-gpu",
            "PageRank (GPU-memory)",
            "examples/pr-gpu/agile_demo_pr_gpu",
            {"-i", "test/1gb.info.txt"}
        }},
        {"reg-report", {
            "reg-report",
            "Kernel Register Report",
            "examples/reg-report/agile_demo_reg_report",
            {}
        }},
    };
    return examples;
}
