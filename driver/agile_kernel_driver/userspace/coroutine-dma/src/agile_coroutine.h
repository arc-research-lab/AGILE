#pragma once

#define __cpp_lib_coroutine
#include <coroutine>
#include <utility>
#include <chrono>

struct DmaRequest {
    struct this_coro_t {};
    struct promise_type {
        uint64_t coroutine_idx;
        std::coroutine_handle<promise_type> self;
        std::chrono::high_resolution_clock::time_point t0;
        std::chrono::high_resolution_clock::time_point t1;
        std::chrono::high_resolution_clock::time_point t2;
        std::chrono::high_resolution_clock::time_point t3;
        std::chrono::high_resolution_clock::time_point t4;
        std::chrono::high_resolution_clock::time_point t5;
        std::chrono::high_resolution_clock::time_point t6;
        std::chrono::high_resolution_clock::time_point t7;
        std::chrono::high_resolution_clock::time_point t8;
        std::chrono::high_resolution_clock::time_point t9;

        // Suspend the coroutine immediately after the coroutine is created.
        // The coroutine is created in the monitoring threads, and it will be executed later in the worker threads.
        std::suspend_always initial_suspend() {
            return {};
        }

        std::suspend_never final_suspend() noexcept {
            return {};
        }

        // This will be called when the DMA transfer is complete
        void await_resume() {
            // Handle the completion of the DMA transfer
        }

        void return_void() {
            // Handle the completion of the coroutine
        }

        DmaRequest get_return_object() {
            self = std::coroutine_handle<promise_type>::from_promise(*this);
            return DmaRequest{self};
        }

        void unhandled_exception() {
            std::cerr << "Unhandled exception in DMA coroutine\n";
            std::terminate();
        }

        auto await_transform(this_coro_t t) noexcept {
            struct awaiter {
                promise_type* p;
                // We don't actually need to suspend to get the handle.
                bool await_ready() const noexcept { return true; }
                void await_suspend(std::coroutine_handle<>) const noexcept {}
                handle_type await_resume() const noexcept {
                    return handle_type::from_promise(*p);
                }
            };
            return awaiter{ this };
        }

        template <class T>
        T&& await_transform(T&& x) noexcept {
            return static_cast<T&&>(x);
        }
    };
    using handle_type = std::coroutine_handle<promise_type>;
    handle_type handle{}; 
    promise_type promise;
};