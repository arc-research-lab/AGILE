#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <CLI/CLI.hpp>
#include <logger.hpp>

#include "demo_ipc.h"
#include "demo_examples.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace asio = boost::asio;
namespace beast = boost::beast;
namespace websocket = beast::websocket;
using tcp = asio::ip::tcp;

namespace fs = std::filesystem;

constexpr uint16_t kDefaultPort = 9002;

/* ── tiny JSON helpers (no external dependency) ─────────────────── */

/* Extract the string value for a given key from a flat JSON object.
   Good enough for  {"action":"run","example":"hello-world"}         */
std::string json_get(const std::string& json, const std::string& key) {
	const std::string needle = "\"" + key + "\"";
	auto pos = json.find(needle);
	if (pos == std::string::npos) return {};

	pos = json.find(':', pos + needle.size());
	if (pos == std::string::npos) return {};

	pos = json.find('"', pos + 1);
	if (pos == std::string::npos) return {};

	auto end = json.find('"', pos + 1);
	if (end == std::string::npos) return {};

	return json.substr(pos + 1, end - pos - 1);
}

std::string json_msg(const std::string& type, const std::string& example,
                     const std::string& data) {
	/* Escape special characters in data for JSON safety */
	std::string escaped;
	escaped.reserve(data.size() + 16);
	for (char c : data) {
		switch (c) {
			case '"':  escaped += "\\\""; break;
			case '\\': escaped += "\\\\"; break;
			case '\n': escaped += "\\n";  break;
			case '\r': escaped += "\\r";  break;
			case '\t': escaped += "\\t";  break;
			default:   escaped += c;      break;
		}
	}
	return "{\"type\":\"" + type + "\",\"example\":\"" + example +
	       "\",\"data\":\"" + escaped + "\"}";
}

fs::path resolve_build_dir(fs::path build_dir) {
	if (build_dir.empty()) return build_dir;

	std::error_code ec;
	fs::path normalized = fs::weakly_canonical(build_dir, ec);
	if (ec) normalized = build_dir;

	// if (fs::exists(normalized / "examples")) return normalized;
	if (fs::exists(normalized / "build" / "examples")) return normalized / "build";

	return normalized;
}

/* ── WebSocket send helper (thread-safe) ────────────────────────── */

using ws_stream = websocket::stream<tcp::socket>;

struct WsSession {
	ws_stream&  ws;
	std::mutex  mu;

	explicit WsSession(ws_stream& s) : ws(s) {}

	void send(const std::string& text) {
		std::lock_guard<std::mutex> lk(mu);
		ws.text(true);
		ws.write(asio::buffer(text));
	}
};

/* ── run an example ─────────────────────────────────────────────── */

void run_example(WsSession& session, const DemoExample& ex,
                 const std::string& build_dir, const std::string& extra_args)
{
	const std::string queue_name = "agile_demo_" + ex.id;
	fs::path resolved_build_dir = resolve_build_dir(build_dir);
	fs::path bin_path = resolved_build_dir / ex.binary;

	if (!fs::exists(bin_path)) {
		session.send(json_msg("error", ex.id,
				"Binary not found: " + bin_path.string()));
		return;
	}

	/* Create the IPC queue (server owns it). */
	demo_ipc::QueueOwner queue(queue_name);

	session.send(json_msg("start", ex.id,
			"Launching " + ex.display_name + " ..."));

	/* Build the command line.  cd into build_dir so relative paths
	   in default_args (e.g. test/bfs_verify.info.txt) resolve. */
	std::ostringstream cmd;
	cmd << "cd " << resolved_build_dir.string() << " && " << bin_path.string();
	for (const auto& a : ex.default_args) cmd << " " << a;
	cmd << " --ipc-queue " << queue_name;
	if (!extra_args.empty()) cmd << " " << extra_args;

	LOG_INFO("SERVER", "Exec: %s", cmd.str().c_str());

	/* Spawn in a detached thread so we can read IPC concurrently. */
	std::atomic<bool> proc_done{false};
	std::thread proc_thread([&]() {
		int rc = std::system(cmd.str().c_str());
		(void)rc;
		proc_done.store(true);
	});

	/* Read messages from the queue until done sentinel or timeout. */
	while (true) {
		std::string msg;
		auto deadline = boost::posix_time::microsec_clock::universal_time()
						+ boost::posix_time::seconds(2);
		if (queue.try_receive(msg, deadline)) {
			if (demo_ipc::is_done(msg)) break;
			session.send(json_msg("output", ex.id, msg));
		} else {
			/* Timeout — check if the process crashed. */
			if (proc_done.load()) {
				session.send(json_msg("output", ex.id,
					"(example process exited without sending done signal)"));
				break;
			}
		}
	}

	if (proc_thread.joinable()) proc_thread.join();

	session.send(json_msg("done", ex.id,
			ex.display_name + " finished."));
}

fs::path ctc_saved_results_path(const std::string& build_dir) {
	fs::path resolved = resolve_build_dir(build_dir);
	fs::path demo_dir = resolved.parent_path();
	return demo_dir / "html" / "data" / "sweep_ctc_data.json";
}

fs::path reg_report_saved_results_path(const std::string& build_dir) {
	fs::path resolved = resolve_build_dir(build_dir);
	fs::path demo_dir = resolved.parent_path();
	return demo_dir / "html" / "data" / "reg_report_data.json";
}

void replay_saved_ctc_sweep(WsSession& session, const std::string& build_dir) {
	fs::path save_path = ctc_saved_results_path(build_dir);
	if (!fs::exists(save_path)) {
		session.send(json_msg("error", "ctc-sweep",
				"Saved CTC results not found: " + save_path.string()));
		return;
	}

	std::ifstream input(save_path);
	if (!input) {
		session.send(json_msg("error", "ctc-sweep",
				"Failed to open saved CTC results: " + save_path.string()));
		return;
	}

	std::string contents((std::istreambuf_iterator<char>(input)),
	                    std::istreambuf_iterator<char>());
	static const std::regex row_re("\\{[^\\{\\}]*\"compute_itr\"[^\\{\\}]*\\}");

	session.send(json_msg("start", "ctc-sweep",
			"Loading saved CTC results from " + save_path.string()));

	auto begin = std::sregex_iterator(contents.begin(), contents.end(), row_re);
	auto end = std::sregex_iterator();
	int count = 0;
	for (auto it = begin; it != end; ++it) {
		std::string msg = "{\"type\":\"bench-result\",\"example\":\"ctc-sweep\",\"data\":"
		                + it->str() + "}";
		session.send(msg);
		++count;
	}

	if (count == 0) {
		session.send(json_msg("error", "ctc-sweep",
				"No saved CTC results found in " + save_path.string()));
		return;
	}

	session.send(json_msg("done", "ctc-sweep",
			"Loaded " + std::to_string(count) + " saved CTC results."));
}

/* ── run CTC sweep ───────────────────────────────────────────────── */

void run_ctc_sweep(WsSession& session, const std::string& build_dir,
                   const std::string& bdf)
{
	fs::path resolved = resolve_build_dir(build_dir);
	fs::path bin_path = resolved / "examples" / "ctc" / "agile_demo_ctc";
	fs::path save_path = ctc_saved_results_path(build_dir);

	if (!fs::exists(bin_path)) {
		session.send(json_msg("error", "ctc-sweep",
				"Binary not found: " + bin_path.string()));
		return;
	}

	/* Default BDF; override via {"action":"ctc-sweep","bdf":"0000:e2:00.0"} */
	std::string bdf_val = bdf.empty() ? "0000:01:00.0" : bdf;

	/* Basic BDF format validation (DDDD:DD:DD.D) */
	{
		static const std::regex bdf_re("^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\\.[0-9]$");
		if (!std::regex_match(bdf_val, bdf_re)) {
			session.send(json_msg("error", "ctc-sweep",
					"Invalid BDF format: " + bdf_val +
					" (expected DDDD:DD:DD.D e.g. 0000:01:00.0)"));
			return;
		}
	}

	const std::string dev = "/dev/AGILE-NVMe-" + bdf_val;
	std::vector<std::string> saved_rows;
	saved_rows.reserve(((2000 - 20) / 20) + 1);

	std::error_code ec;
	fs::create_directories(save_path.parent_path(), ec);
	if (ec) {
		session.send(json_msg("error", "ctc-sweep",
				"Failed to prepare save directory for " + save_path.string() + ": " + ec.message()));
		return;
	}

	if (fs::exists(save_path)) {
		fs::remove(save_path, ec);
		if (ec) {
			session.send(json_msg("error", "ctc-sweep",
					"Failed to delete stale CTC results at " + save_path.string() + ": " + ec.message()));
			return;
		}
	}

	session.send(json_msg("start", "ctc-sweep",
			"Launching CTC sweep on " + dev + " (compute_itr 20..2000, step 20) ..."));
	const int itr_start = 20, itr_end = 2000, itr_step = 20;

	for (int citr = itr_start; citr <= itr_end; citr += itr_step) {
		std::ostringstream cmd;
		cmd << bin_path.string()
		    << " -d " << dev
		    << " -q 2 --queue-depth 512 --buf-per-blk 2"
		    << " -i 1000 -t 32"
		    << " --compute-itr " << citr
		    << " 2>&1";

		LOG_INFO("CTC-SWEEP", "[%d/%d] compute_itr=%d",
				(citr - itr_start) / itr_step + 1,
				(itr_end - itr_start) / itr_step + 1, citr);

		FILE* fp = popen(cmd.str().c_str(), "r");
		if (!fp) {
			session.send(json_msg("error", "ctc-sweep",
					"Failed to launch CTC binary"));
			return;
		}

		std::string sync_time, async_time, ctc_val, speedup_val;
		char line[4096];
		while (fgets(line, sizeof(line), fp)) {
			std::string s(line);
			while (!s.empty() && (s.back() == '\n' || s.back() == '\r'))
				s.pop_back();
			if (s.empty()) continue;
			LOG_INFO("CTC", "%s", s.c_str());

			/* Parse summary lines. */
			if (s.find("Sync (L+C)") != std::string::npos) {
				/* "[INFO][CTC]      Sync (L+C):   266704 us" */
				auto pos = s.find("Sync (L+C):");
				if (pos != std::string::npos) {
					std::istringstream iss(s.substr(pos));
					std::string w1, w2;
					iss >> w1 >> w2 >> sync_time;  /* "Sync" "(L+C):" "266704" */
				}
			} else if (s.find("Async (L+C)") != std::string::npos) {
				auto pos = s.find("Async (L+C):");
				if (pos != std::string::npos) {
					std::istringstream iss(s.substr(pos));
					std::string w1, w2;
					iss >> w1 >> w2 >> async_time;
				}
			} else if (s.find("CTC (Compute-to-Communication)") != std::string::npos) {
				auto pos = s.find("CTC (Compute-to-Communication):");
				if (pos != std::string::npos) {
					std::istringstream iss(s.substr(pos));
					std::string w1, w2;
					iss >> w1 >> w2 >> ctc_val;
				}
			} else if (s.find("Async speedup over sync") != std::string::npos) {
				auto pos = s.find("Async speedup over sync:");
				if (pos != std::string::npos) {
					std::istringstream iss(s.substr(pos));
					std::string w1, w2, w3, w4;
					iss >> w1 >> w2 >> w3 >> w4 >> speedup_val;
					/* Remove trailing 'x' */
					if (!speedup_val.empty() && speedup_val.back() == 'x')
						speedup_val.pop_back();
				}
			}
		}

		int rc = pclose(fp);
		if (rc != 0) {
			LOG_ERROR("CTC-SWEEP", "compute_itr=%d exited with %d", citr, WEXITSTATUS(rc));
		}

		/* Send per-step JSON result. */
		std::string row = "{\"compute_itr\":" + std::to_string(citr) +
		                  ",\"sync_time\":" + sync_time +
		                  ",\"async_time\":" + async_time +
		                  ",\"ctc\":" + ctc_val +
		                  ",\"speedup\":" + speedup_val + "}";
		saved_rows.push_back(row);
		std::string msg = "{\"type\":\"bench-result\",\"example\":\"ctc-sweep\",\"data\":"
		                + row + "}";
		session.send(msg);
	}

	std::ofstream output(save_path, std::ios::trunc);
	if (!output) {
		session.send(json_msg("error", "ctc-sweep",
				"Failed to save CTC results to " + save_path.string()));
	} else {
		output << "{\n  \"sweep_ctc_02\": [\n";
		for (std::size_t index = 0; index < saved_rows.size(); ++index) {
			if (index != 0) output << ",\n";
			output << "    " << saved_rows[index];
		}
		output << "\n  ]\n}\n";
	}

	session.send(json_msg("done", "ctc-sweep", "CTC sweep finished."));
}

/* ── run sweep benchmark ─────────────────────────────────────────── */

void run_sweep(WsSession& session, const std::string& build_dir,
               const std::string& extra_args)
{
	fs::path resolved = resolve_build_dir(build_dir);
	/* The script lives at demo/scripts/sweep_bench.sh.
	   build_dir is typically demo/build, so demo = build_dir/..       */
	fs::path demo_dir = resolved.parent_path();
	fs::path script   = demo_dir / "scripts" / "sweep_bench.sh";

	if (!fs::exists(script)) {
		session.send(json_msg("error", "sweep",
				"Script not found: " + script.string()));
		return;
	}

	session.send(json_msg("start", "sweep", "Launching sweep benchmark ..."));

	std::ostringstream cmd;
	cmd << "cd " << demo_dir.string() << " && bash " << script.string()
	    << " --skip-write";
	if (!extra_args.empty()) cmd << " " << extra_args;
	cmd << " 2>&1";

	LOG_INFO("SERVER", "Sweep exec: %s", cmd.str().c_str());

	FILE* fp = popen(cmd.str().c_str(), "r");
	if (!fp) {
		session.send(json_msg("error", "sweep", "Failed to launch sweep script"));
		return;
	}

	char line[4096];
	const std::string json_marker = "__JSON_RESULT__";
	while (fgets(line, sizeof(line), fp)) {
		/* Strip trailing newline. */
		std::string s(line);
		while (!s.empty() && (s.back() == '\n' || s.back() == '\r'))
			s.pop_back();
		if (s.empty()) continue;

		/* Lines starting with __JSON_RESULT__ carry per-test JSON. */
		if (s.rfind(json_marker, 0) == 0) {
			std::string json_data = s.substr(json_marker.size());
			std::string msg = "{\"type\":\"bench-result\",\"example\":\"sweep\",\"data\":"
			                  + json_data + "}";
			session.send(msg);
		} else {
			LOG_INFO("SWEEP", "%s", s.c_str());
		}
	}

	int rc = pclose(fp);

	if (rc != 0) {
		session.send(json_msg("error", "sweep",
				"Sweep finished with exit code " + std::to_string(WEXITSTATUS(rc))));
	}

	session.send(json_msg("done", "sweep", "Sweep benchmark finished."));
}

/* ── register report ─────────────────────────────────────────── */

void replay_saved_reg_report(WsSession& session, const std::string& build_dir) {
	fs::path save_path = reg_report_saved_results_path(build_dir);
	if (!fs::exists(save_path)) {
		session.send(json_msg("error", "reg-report",
				"Saved register report not found: " + save_path.string()));
		return;
	}

	std::ifstream input(save_path);
	if (!input) {
		session.send(json_msg("error", "reg-report",
				"Failed to open saved register report: " + save_path.string()));
		return;
	}

	std::string contents((std::istreambuf_iterator<char>(input)),
	                    std::istreambuf_iterator<char>());
	if (contents.empty()) {
		session.send(json_msg("error", "reg-report",
				"Saved register report is empty: " + save_path.string()));
		return;
	}

	session.send(json_msg("start", "reg-report",
			"Loading saved register report from " + save_path.string()));
	session.send("{\"type\":\"reg-report\",\"data\":" + contents + "}");
	session.send(json_msg("done", "reg-report", "Loaded saved register report."));
}

void run_reg_report(WsSession& session, const std::string& build_dir) {
	fs::path resolved = resolve_build_dir(build_dir);
	fs::path bin = resolved / "examples" / "reg-report" / "agile_demo_reg_report";
	fs::path save_path = reg_report_saved_results_path(build_dir);

	if (!fs::exists(bin)) {
		session.send(json_msg("error", "reg-report",
				"Binary not found: " + bin.string()));
		return;
	}

	session.send(json_msg("start", "reg-report", "Running register report..."));

	std::string cmd = bin.string() + " 2>&1";
	LOG_INFO("REG-REPORT", "Exec: %s", cmd.c_str());

	FILE* fp = popen(cmd.c_str(), "r");
	if (!fp) {
		session.send(json_msg("error", "reg-report",
				"Failed to launch reg-report binary"));
		return;
	}

	/* Collect rows as JSON array. */
	std::ostringstream rows;
	rows << "[";
	bool first = true;
	char line[4096];

	/* Strip ANSI escape sequences (e.g. \x1b[1;33m). */
	std::regex ansi_re("\x1b\\[[0-9;]*m");
	auto strip_ansi = [&](const std::string& in) {
		return std::regex_replace(in, ansi_re, "");
	};

	while (fgets(line, sizeof(line), fp)) {
		std::string s(line);
		while (!s.empty() && (s.back() == '\n' || s.back() == '\r'))
			s.pop_back();
		if (s.empty()) continue;
		LOG_INFO("REG-REPORT", "%s", s.c_str());

		/* Strip ANSI codes before parsing. */
		s = strip_ansi(s);

		/* Data lines contain │ (UTF-8 box-drawing). */
		auto sep = s.find("\xe2\x94\x82");   /* UTF-8 for │ */
		if (sep == std::string::npos) continue;

		std::string left  = s.substr(0, sep);
		std::string right = s.substr(sep + 3); /* 3-byte UTF-8 char */

		/* Trim whitespace. */
		auto trim = [](std::string& t) {
			while (!t.empty() && std::isspace((unsigned char)t.front())) t.erase(t.begin());
			while (!t.empty() && std::isspace((unsigned char)t.back()))  t.pop_back();
		};
		trim(left);
		trim(right);

		/* right looks like: "bfs_kernel: 48 regs, 396 B cmem" */
		std::string kernel, regs_str, cmem_str;
		auto colon = right.find(':');
		if (colon != std::string::npos) {
			kernel = right.substr(0, colon);
			trim(kernel);
			std::string rest = right.substr(colon + 1);
			trim(rest);

			/* Extract regs count. */
			std::regex re_regs("(\\d+)\\s+regs");
			std::smatch m;
			if (std::regex_search(rest, m, re_regs))
				regs_str = m[1].str();

			/* Extract cmem. */
			std::regex re_cmem("(\\d+)\\s+B\\s+cmem");
			if (std::regex_search(rest, m, re_cmem))
				cmem_str = m[1].str();
		}

		if (!first) rows << ",";
		first = false;
		rows << "{\"label\":\"" << left
		     << "\",\"kernel\":\"" << kernel
		     << "\",\"regs\":" << (regs_str.empty() ? "0" : regs_str)
		     << ",\"cmem\":" << (cmem_str.empty() ? "0" : cmem_str)
		     << "}";
	}

	rows << "]";

	int rc = pclose(fp);
	if (rc != 0) {
		session.send(json_msg("error", "reg-report",
				"reg-report exited with code " + std::to_string(WEXITSTATUS(rc))));
		return;
	}

	std::error_code ec;
	fs::create_directories(save_path.parent_path(), ec);
	if (ec) {
		session.send(json_msg("error", "reg-report",
				"Failed to prepare save directory for " + save_path.string() + ": " + ec.message()));
		return;
	}

	std::ofstream output(save_path, std::ios::trunc);
	if (!output) {
		session.send(json_msg("error", "reg-report",
				"Failed to save register report to " + save_path.string()));
		return;
	}
	output << rows.str() << "\n";

	/* Send structured result. */
	std::string msg = "{\"type\":\"reg-report\",\"data\":" + rows.str() + "}";
	session.send(msg);
	session.send(json_msg("done", "reg-report", "Register report finished."));
}

/* ── handle one WebSocket client ────────────────────────────────── */

void handle_session(tcp::socket socket, const std::string& build_dir) {
	ws_stream ws(std::move(socket));
	ws.accept();

	WsSession session(ws);

	/* Send welcome + available examples list. */
	{
		std::ostringstream list;
		list << "{\"type\":\"welcome\",\"examples\":[";
		bool first = true;
		for (const auto& [id, ex] : get_demo_examples()) {
			if (!first) list << ",";
			list << "{\"id\":\"" << ex.id << "\",\"name\":\"" << ex.display_name << "\"}";
			first = false;
		}
		list << "]}";
		session.send(list.str());
	}

	for (;;) {
		beast::flat_buffer buffer;
		ws.read(buffer);

		const std::string raw = beast::buffers_to_string(buffer.data());
		LOG_INFO("SERVER", "Received: %s", raw.c_str());

		const std::string action = json_get(raw, "action");

		if (action == "run") {
			const std::string example_id = json_get(raw, "example");
			const std::string extra_args = json_get(raw, "args");
			const auto& registry = get_demo_examples();
			auto it = registry.find(example_id);

			if (it == registry.end()) {
				session.send(json_msg("error", example_id,
						"Unknown example: " + example_id));
				continue;
			}

			run_example(session, it->second, build_dir, extra_args);

		} else if (action == "list") {
			/* Re-send the example list. */
			std::ostringstream list;
			list << "{\"type\":\"examples\",\"examples\":[";
			bool first = true;
			for (const auto& [id, ex] : get_demo_examples()) {
				if (!first) list << ",";
				list << "{\"id\":\"" << ex.id << "\",\"name\":\""
				     << ex.display_name << "\"}";
				first = false;
			}
			list << "]}";
			session.send(list.str());

		} else if (action == "sweep") {
			const std::string extra_args = json_get(raw, "args");
			run_sweep(session, build_dir, extra_args);

		} else if (action == "ctc-sweep") {
			const std::string bdf = json_get(raw, "bdf");
			run_ctc_sweep(session, build_dir, bdf);

		} else if (action == "ctc-load-saved") {
			replay_saved_ctc_sweep(session, build_dir);

		} else if (action == "reg-report") {
			run_reg_report(session, build_dir);

		} else if (action == "reg-report-load-saved") {
			replay_saved_reg_report(session, build_dir);

		} else {
			/* Echo fallback for plain text messages */
			session.send(json_msg("echo", "", raw));
		}
	}
}

}  // namespace

int main(int argc, char** argv) {
	uint16_t port = kDefaultPort;
	std::string build_dir;

	CLI::App app{"Boost.Beast WebSocket demo server"};
	app.add_option("-p,--port", port, "TCP port to listen on")
			->check(CLI::Range(1, 65535));
	app.add_option("-b,--build-dir", build_dir,
			"Path to the CMake build directory (contains examples/)")
			->default_val("");
	CLI11_PARSE(app, argc, argv);

	/* Auto-detect build dir: look for ../build relative to the binary. */
	if (build_dir.empty()) {
		fs::path exe_path = fs::canonical("/proc/self/exe");
		/* Binary is usually at build/examples/server/websocket_demo_server */
		fs::path candidate = exe_path.parent_path().parent_path().parent_path();
		if (fs::exists(candidate / "examples")) {
			build_dir = candidate.string();
		} else {
			build_dir = ".";
		}
	}

	build_dir = resolve_build_dir(build_dir).string();

	LOG_INFO("SERVER", "Build directory: %s", build_dir.c_str());

	try {
		asio::io_context io_context(1);
		tcp::acceptor acceptor(io_context, {tcp::v4(), port});

		LOG_INFO("SERVER", "WebSocket server listening on ws://0.0.0.0:%d", port);
		LOG_INFO("SERVER", "Open demo/html/index.html in a browser and connect.");

		while (true) {
			tcp::socket socket(io_context);
			acceptor.accept(socket);

			const auto remote_ep = socket.remote_endpoint();
			LOG_INFO("SERVER", "Client connected: %s:%d",
					 remote_ep.address().to_string().c_str(),
					 remote_ep.port());

			try {
				handle_session(std::move(socket), build_dir);
			} catch (const beast::system_error& error) {
				if (error.code() != websocket::error::closed) {
					LOG_ERROR("SERVER", "session error: %s", error.what());
				}
			} catch (const std::exception& error) {
				LOG_ERROR("SERVER", "session error: %s", error.what());
			}

			LOG_INFO("SERVER", "Client disconnected");
		}
	} catch (const std::exception& error) {
		LOG_ERROR("SERVER", "server error: %s", error.what());
		return 1;
	}

	return 0;
}
