/*
 * reg-report  –  Compile AGILE & BAM BFS/PR kernels with -Xptxas -v, 
 *                parse register-per-thread, and report via IPC.
 *
 * This is a pure C++ tool (no CUDA runtime needed).  It shells out to
 * nvcc --device-c and scrapes the ptxas verbose output.
 */

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>

/* We only need the QueueClient (example side) from demo_ipc.h.
   Define __CUDACC__ to skip the server-only QueueOwner that needs
   boost::posix_time.  */
#define __CUDACC__
#include "demo_ipc.h"
#undef __CUDACC__

namespace fs = std::filesystem;

/* ── Run a command and capture both stdout+stderr ────────────── */
static std::string exec_cmd(const std::string &cmd) {
    std::string result;
    std::string full = cmd + " 2>&1";
    FILE *fp = popen(full.c_str(), "r");
    if (!fp) return "(failed to run command)";
    char buf[4096];
    while (fgets(buf, sizeof(buf), fp))
        result += buf;
    pclose(fp);
    return result;
}

/* ── Demangle a kernel name (best-effort, falls back to mangled) ─ */
static std::string demangle(const std::string &mangled) {
    std::string cmd = "c++filt '" + mangled + "' 2>/dev/null";
    std::string out = exec_cmd(cmd);
    /* strip trailing newline */
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r'))
        out.pop_back();
    if (out.empty() || out == mangled) return mangled;
    return out;
}

/* ── Extract short kernel name from demangled signature ────────── */
static std::string short_name(const std::string &demangled) {
    /* Try to get just the function name before '(' or '<'
       e.g. "void start_agile_cq_service<...>(...)" → "start_agile_cq_service"
            "void bfs_kernel(AgileCtrl<...>*, ...)"  → "bfs_kernel"            */
    /* Find first '<' or '(' — whichever comes first is the end of the base name */
    auto lt = demangled.find('<');
    auto lp = demangled.find('(');
    std::string::size_type end = std::string::npos;
    if (lt != std::string::npos && lp != std::string::npos)
        end = std::min(lt, lp);
    else if (lt != std::string::npos)
        end = lt;
    else if (lp != std::string::npos)
        end = lp;
    std::string prefix = (end != std::string::npos) ? demangled.substr(0, end) : demangled;
    /* Take the last word (after the last space) */
    auto sp = prefix.rfind(' ');
    if (sp != std::string::npos)
        prefix = prefix.substr(sp + 1);
    while (!prefix.empty() && prefix.front() == ' ') prefix.erase(prefix.begin());
    while (!prefix.empty() && prefix.back() == ' ') prefix.pop_back();
    if (prefix.empty()) return demangled;
    return prefix;
}

/* ── ANSI color helpers ─────────────────────────────────────── */
namespace clr {
    constexpr const char *reset   = "\033[0m";
    constexpr const char *bold    = "\033[1m";
    constexpr const char *dim     = "\033[2m";
    constexpr const char *green   = "\033[32m";
    constexpr const char *yellow  = "\033[33m";
    constexpr const char *cyan    = "\033[36m";
    constexpr const char *bgreen  = "\033[1;32m";
    constexpr const char *byellow = "\033[1;33m";
    constexpr const char *bcyan   = "\033[1;36m";
}

struct KernelInfo {
    std::string mangled;
    std::string display;      /* short, human-readable */
    int registers = 0;
    int cmem = 0;
    int smem = 0;
    int stack = 0;
    int spill_stores = 0;
    int spill_loads = 0;
};

/* ── Parse ptxas verbose output ────────────────────────────────── */
static std::vector<KernelInfo> parse_ptxas(const std::string &output) {
    std::vector<KernelInfo> results;

    /* Patterns:
       ptxas info    : Compiling entry function '_Z10bfs_kernel...' for 'sm_89'
       ptxas info    : Used 50 registers, 396 bytes cmem[0]
       ptxas info    : Function properties for _Z10bfs_kernel...
           104 bytes stack frame, 52 bytes spill stores, 52 bytes spill loads
    */
    std::regex re_entry(R"(Compiling entry function '([^']+)')");
    std::regex re_used(R"(Used (\d+) registers(?:, (\d+) bytes smem)?(?:.*?(\d+) bytes cmem)?)");
    std::regex re_func_props(R"(Function properties for (\S+))");
    std::regex re_stack(R"((\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads)");

    std::istringstream iss(output);
    std::string line;
    KernelInfo *current = nullptr;
    std::string last_func_props_name;

    while (std::getline(iss, line)) {
        std::smatch m;

        if (std::regex_search(line, m, re_entry)) {
            results.push_back({});
            current = &results.back();
            current->mangled = m[1].str();
            std::string dm = demangle(current->mangled);
            current->display = short_name(dm);
            continue;
        }

        if (current && std::regex_search(line, m, re_used)) {
            current->registers = std::stoi(m[1].str());
            if (m[2].matched) current->smem = std::stoi(m[2].str());
            if (m[3].matched) current->cmem = std::stoi(m[3].str());
            continue;
        }

        /* Track function properties for entry functions (spill info) */
        if (std::regex_search(line, m, re_func_props)) {
            last_func_props_name = m[1].str();
            continue;
        }

        if (std::regex_search(line, m, re_stack)) {
            /* Match back to kernel if name matches */
            for (auto &ki : results) {
                if (ki.mangled == last_func_props_name) {
                    ki.stack = std::stoi(m[1].str());
                    ki.spill_stores = std::stoi(m[2].str());
                    ki.spill_loads = std::stoi(m[3].str());
                    break;
                }
            }
            last_func_props_name.clear();
            continue;
        }
    }

    return results;
}

/* ── A compilation target definition ───────────────────────────── */
struct CompileTarget {
    std::string label;          /* e.g. "BFS (AGILE)" */
    std::string source;         /* relative to AGILE_ROOT/demo/ */
    std::vector<std::string> extra_includes;  /* relative to AGILE_ROOT */
    std::vector<std::string> extra_flags;
    /* Which kernel names to highlight (substring match); empty = show all */
    std::vector<std::string> kernel_filter;
};


int main(int argc, char **argv) {
    CLI::App app{"Register Report — compile kernels and report register usage"};

    std::string ipc_queue;
    app.add_option("--ipc-queue", ipc_queue, "IPC message queue name");

    std::string agile_root;
    app.add_option("--agile-root", agile_root, "AGILE repository root");

    std::string nvcc_path = "nvcc";
    app.add_option("--nvcc", nvcc_path, "Path to nvcc");

    std::string arch = "native";
    app.add_option("--arch", arch, "CUDA architecture");

    CLI11_PARSE(app, argc, argv);

    /* Auto-detect AGILE_ROOT from binary location */
    if (agile_root.empty()) {
        auto exe = fs::canonical("/proc/self/exe");
        /* binary is at <root>/demo/build/examples/reg-report/agile_demo_reg_report */
        agile_root = exe.parent_path().parent_path().parent_path()
                         .parent_path().parent_path().string();
    }

    std::string demo_root = agile_root + "/demo";
    std::string cli11_inc = demo_root + "/build/_deps/cli11-src/include";

    /* Common include dirs (relative to agile_root) */
    std::vector<std::string> agile_incs = {
        agile_root + "/include",
        agile_root + "/benchmarks/common",
        agile_root + "/common",
        demo_root + "/examples/common",
        cli11_inc,
    };
    std::vector<std::string> bam_incs = {
        agile_root + "/baseline/bam/include",
        agile_root + "/baseline/bam/include/freestanding/include",
    };

    /* Define compilation targets */
    std::vector<CompileTarget> targets = {
        {
            "BFS (AGILE)",
            "examples/bfs-agile/src/main.cu",
            {},   /* only agile_incs */
            {},
            {"bfs_kernel"},
        },
        {
            "BFS (BAM)",
            "examples/bfs-bam/src/main.cu",
            bam_incs,
            {"-Xcompiler=-fpermissive"},
            {"bfs_kernel_bam"},
        },
        {
            "PageRank (AGILE)",
            "examples/pr-agile/src/main.cu",
            {},
            {},
            {"pagerank_kernel"},
        },
        {
            "PageRank (BAM)",
            "examples/pr-bam/src/main.cu",
            bam_incs,
            {"-Xcompiler=-fpermissive"},
            {"pagerank_kernel_bam"},
        },
    };

    /* ── Helper: check if a ptxas line mentions a filtered kernel ── */
    auto line_mentions_kernel = [](const std::string &line,
                                   const std::vector<std::string> &filters) -> bool {
        for (const auto &f : filters)
            if (line.find(f) != std::string::npos) return true;
        return false;
    };

    /* IPC collector for the final summary only (plain text without ANSI) */
    std::vector<std::string> ipc_messages;

    /* Helper: print colored line to stdout immediately */
    auto emit = [&](const std::string &plain, const std::string &colored) {
        std::cout << colored << "\n";
    };
    auto emit_plain = [&](const std::string &msg) {
        std::cout << msg << "\n";
    };

    emit("=== Kernel Register Report ===",
         std::string(clr::bcyan) + "=== Kernel Register Report ===" + clr::reset);
    emit_plain("");

    /* Collect summary entries for the table at the end */
    struct SummaryEntry { std::string label; std::string kernel; int regs; int cmem; };
    std::vector<SummaryEntry> summary;

    for (const auto &tgt : targets) {
        std::string src = demo_root + "/" + tgt.source;
        if (!fs::exists(src)) {
            std::string msg = "[" + tgt.label + "] source not found: " + tgt.source;
            emit(msg, std::string(clr::yellow) + msg + clr::reset);
            continue;
        }

        /* ── Section header (printed BEFORE compilation) ── */
        std::string header = "── " + tgt.label + " ── (" + tgt.source + ")";
        emit(header, std::string(clr::bcyan) + header + clr::reset);
        std::cout << std::flush;

        /* Build nvcc command */
        std::ostringstream cmd;
        cmd << nvcc_path
            << " -std=c++17 --expt-relaxed-constexpr -w"
            << " -Xptxas -v"
            << " -arch=" << arch
            << " -DENABLE_LOGGING=0"
            << " --device-c";

        for (const auto &inc : agile_incs)
            cmd << " -I" << inc;
        for (const auto &inc : tgt.extra_includes)
            cmd << " -I" << inc;
        for (const auto &flag : tgt.extra_flags)
            cmd << " " << flag;

        cmd << " " << src << " -o /tmp/reg_report_tmp.o";

        std::string command = cmd.str();
        fprintf(stderr, "Compiling: %s\n", tgt.label.c_str());

        std::string output = exec_cmd(command);
        auto kernels = parse_ptxas(output);

        /* Filter to relevant kernels */
        std::vector<KernelInfo> filtered;
        for (const auto &ki : kernels) {
            for (const auto &filt : tgt.kernel_filter) {
                if (ki.mangled.find(filt) != std::string::npos ||
                    ki.display.find(filt) != std::string::npos) {
                    filtered.push_back(ki);
                    break;
                }
            }
        }

        /* ── Full compile log with highlights (printed immediately) ── */
        {
            std::istringstream iss(output);
            std::string line;
            bool highlight_until_used = false;
            while (std::getline(iss, line)) {
                if (line.empty()) continue;
                bool highlight = line_mentions_kernel(line, tgt.kernel_filter);
                if (highlight_until_used) {
                    highlight = true;
                    if (line.find("Used") != std::string::npos)
                        highlight_until_used = false;
                }
                if (line.find("Compiling entry") != std::string::npos &&
                    line_mentions_kernel(line, tgt.kernel_filter))
                    highlight_until_used = true;

                if (highlight) {
                    emit(">>> " + line,
                         std::string(clr::bgreen) + ">>> " + line + clr::reset);
                } else {
                    emit("    " + line,
                         std::string(clr::dim) + "    " + line + clr::reset);
                }
            }
        }

        /* ── Filtered kernel register summary (printed immediately) ── */
        if (!filtered.empty()) {
            for (const auto &ki : filtered) {
                std::ostringstream os_plain, os_color;
                os_plain << "  ** " << ki.display << ": "
                         << ki.registers << " registers";
                os_color << clr::byellow << "  ★ " << ki.display << ": "
                         << ki.registers << " registers";
                if (ki.smem > 0) {
                    os_plain << ", " << ki.smem << " B smem";
                    os_color << ", " << ki.smem << " B smem";
                }
                if (ki.cmem > 0) {
                    os_plain << ", " << ki.cmem << " B cmem";
                    os_color << ", " << ki.cmem << " B cmem";
                }
                if (ki.spill_stores > 0 || ki.spill_loads > 0) {
                    os_plain << " (spill: " << ki.spill_stores << "B store, "
                             << ki.spill_loads << "B load)";
                    os_color << " (spill: " << ki.spill_stores << "B store, "
                             << ki.spill_loads << "B load)";
                }
                os_color << clr::reset;
                emit(os_plain.str(), os_color.str());
                summary.push_back({tgt.label, ki.display, ki.registers, ki.cmem});
            }
        } else {
            emit("  (no matching kernels found)",
                 std::string(clr::dim) + "  (no matching kernels found)" + clr::reset);
        }
        emit_plain("");
    }

    /* ── Summary table ── */
    {
        std::ostringstream output;
        std::size_t label_width = 0;
        for (const auto &s : summary)
            label_width = std::max(label_width, s.label.size());

        std::string border  = "════════════════════════════════════════════════════════════════";
        std::string divider = "────────────────────────────────────────────────────────────────";
        emit(border, std::string(clr::cyan) + border + clr::reset);
        //    ipc_messages.push_back(border);
        emit("                Summary: Registers per Thread",
             std::string(clr::bcyan) + "                Summary: Registers per Thread" + clr::reset);
        //    ipc_messages.push_back("                Summary: Registers per Thread");
        emit(divider, std::string(clr::cyan) + divider + clr::reset);
        //    ipc_messages.push_back(divider);
        output << "[";
        for (const auto &s : summary) {
            std::ostringstream os_plain, os_color;
            os_plain << "  " << std::left << std::setw(label_width) << s.label
                     << "  │  " << s.kernel << ": " << s.regs << " regs";
            os_color << clr::bold << "  " << std::left << std::setw(label_width) << s.label << clr::reset
                     << "  │  " << clr::byellow << s.kernel << ": " << s.regs << " regs";
            if (s.cmem > 0) {
                os_plain << ", " << s.cmem << " B cmem";
                os_color << ", " << s.cmem << " B cmem";
            }
            output << "{\"label\": \"" + s.label + "\", \"kernel\": \"" + s.kernel + "\", \"regs\": " + std::to_string(s.regs) + ", \"cmem\": " + std::to_string(s.cmem) + "},";
            os_color << clr::reset;
            emit(os_plain.str(), os_color.str());
            // ipc_messages.push_back(os_plain.str());
        }
        emit(border, std::string(clr::cyan) + border + clr::reset);
        output << "{}]";
        ipc_messages.push_back(output.str());
    }

    /* Send via IPC (plain text, no ANSI codes) */
    if (!ipc_queue.empty()) {
        demo_ipc::QueueClient ipc(ipc_queue);
        for (const auto &msg : ipc_messages)
            ipc.send(msg);
        ipc.send_done();
    }

    /* Cleanup temp file */
    std::remove("/tmp/reg_report_tmp.o");

    return 0;
}
