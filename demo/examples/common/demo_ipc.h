#pragma once
/*
 * demo_ipc.h  –  Thin IPC layer between demo examples and the WebSocket server.
 *
 * Uses Boost.Interprocess message_queue so the examples (which are
 * separate processes) can send variable-length text messages back to
 * the server, which in turn forwards them to the browser via WebSocket.
 *
 * Protocol
 * --------
 *   • The SERVER creates the queue before spawning an example.
 *   • The EXAMPLE opens the queue, sends one or more messages, and
 *     finally sends a sentinel (empty string "" or the constant
 *     DEMO_IPC_DONE) to signal completion.
 *   • The SERVER reads messages in a loop, forwarding each one to the
 *     browser, until it receives the sentinel.
 *
 * Queue naming: "/agile_demo_<example_name>"
 *   The name is passed to the example via --ipc-queue CLI flag.
 *
 * Message capacity: each queue slot is DEMO_IPC_MAX_MSG bytes.
 *   Adjust if an example needs to send larger payloads.
 */

#include <boost/interprocess/ipc/message_queue.hpp>
#include <cstring>
#include <string>
#include <stdexcept>

namespace demo_ipc {

/* Maximum bytes per message.  4 KB is generous for text output. */
inline constexpr std::size_t MAX_MSG_SIZE = 4096;

/* Maximum number of messages that can be buffered in the queue. */
inline constexpr std::size_t MAX_MSG_COUNT = 128;

/* Sentinel string that signals "example is done". */
inline constexpr const char* DONE_SENTINEL = "__DONE__";

namespace bip = boost::interprocess;

// ─── Server side (not compiled by nvcc) ─────────────────────────────

#ifndef __CUDACC__

/* Create (or recreate) a named message queue.  Call before spawning
   the example process.  Returns a unique_ptr-like owner. */
class QueueOwner {
public:
    explicit QueueOwner(const std::string& name)
        : name_(name)
    {
        /* Remove stale queue from a previous crashed run. */
        bip::message_queue::remove(name_.c_str());
        mq_ = std::make_unique<bip::message_queue>(
            bip::create_only, name_.c_str(), MAX_MSG_COUNT, MAX_MSG_SIZE);
    }

    ~QueueOwner() {
        mq_.reset();
        bip::message_queue::remove(name_.c_str());
    }

    /* Blocking receive.  Returns the message text. */
    std::string receive() {
        char buf[MAX_MSG_SIZE];
        bip::message_queue::size_type recvd_size = 0;
        unsigned int priority = 0;
        mq_->receive(buf, MAX_MSG_SIZE, recvd_size, priority);
        return std::string(buf, recvd_size);
    }

    /* Timed receive.  Returns false on timeout. */
    bool try_receive(std::string& out,
                     const boost::posix_time::ptime& abs_time)
    {
        char buf[MAX_MSG_SIZE];
        bip::message_queue::size_type recvd_size = 0;
        unsigned int priority = 0;
        if (mq_->timed_receive(buf, MAX_MSG_SIZE, recvd_size, priority,
                                abs_time))
        {
            out.assign(buf, recvd_size);
            return true;
        }
        return false;
    }

    QueueOwner(const QueueOwner&) = delete;
    QueueOwner& operator=(const QueueOwner&) = delete;

private:
    std::string name_;
    std::unique_ptr<bip::message_queue> mq_;
};

#endif  // __CUDACC__

// ─── Example (client) side ──────────────────────────────────────────

class QueueClient {
public:
    explicit QueueClient(const std::string& name)
        : mq_(bip::open_only, name.c_str())
    {}

    /* Send a text message to the server. */
    void send(const std::string& msg) {
        if (msg.size() > MAX_MSG_SIZE) {
            throw std::length_error("demo_ipc: message exceeds MAX_MSG_SIZE");
        }
        mq_.send(msg.data(), msg.size(), /*priority=*/0);
    }

    /* Convenience: send the "done" sentinel. */
    void send_done() {
        send(DONE_SENTINEL);
    }

private:
    bip::message_queue mq_;
};

/* Helper: check whether a received message is the done sentinel. */
inline bool is_done(const std::string& msg) {
    return msg == DONE_SENTINEL;
}

}  // namespace demo_ipc
