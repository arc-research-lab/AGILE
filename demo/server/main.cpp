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
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <mutex>
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
	const std::string bin_path = build_dir + "/" + ex.binary;

	if (!fs::exists(bin_path)) {
		session.send(json_msg("error", ex.id,
				"Binary not found: " + bin_path));
		return;
	}

	/* Create the IPC queue (server owns it). */
	demo_ipc::QueueOwner queue(queue_name);

	session.send(json_msg("start", ex.id,
			"Launching " + ex.display_name + " ..."));

	/* Build the command line.  cd into build_dir so relative paths
	   in default_args (e.g. test/bfs_verify.info.txt) resolve. */
	std::ostringstream cmd;
	cmd << "cd " << build_dir << " && " << bin_path;
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
