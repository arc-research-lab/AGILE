#pragma once

#define __cpp_lib_coroutine
#include <coroutine>
#include <utility>
#include <chrono>


#include <time.h>

struct DmaRequest {
    struct this_coro_t {};
    struct promise_type {
        uint64_t coroutine_idx;
        std::coroutine_handle<promise_type> self;
        
        uint64_t dma_callback_time = 0;
        struct timespec init_time;
        struct timespec finish_issue_time;
        struct timespec after_dma_awake_time;
        struct timespec end_time;


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
                auto await_resume() const noexcept {
                    return p;
                }
            };
            return awaiter{ this };
        }

        template <class T>
        T&& await_transform(T&& x) noexcept {
            return static_cast<T&&>(x);
        }
    };

    promise_type * getPromise(){
        return &handle.promise();
    }
    
    using handle_type = std::coroutine_handle<promise_type>;
    handle_type handle{}; 
};