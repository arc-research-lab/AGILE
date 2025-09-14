#pragma once
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string_view>
#include <thread>
#include <utility>
#include <mutex>
#include <atomic>
#if __has_include(<source_location>)
  #include <source_location>
  using src_loc = std::source_location;
#else
  // Fallback shim (won't have function name)
  struct src_loc {
    static src_loc current() { return {}; }
    const char* file_name()  const { return "unknown"; }
    unsigned int line() const { return 0; }
  };
#endif

#if __has_include(<syncstream>) && (__cpp_lib_syncbuf >= 201803L)
  #include <syncstream>
  #define MTLOG_HAS_OSYNCSTREAM 1
#else
  #define MTLOG_HAS_OSYNCSTREAM 0
#endif

namespace mtlog {

enum class level { trace, debug, info, warn, error, critical };

inline const char* level_name(level lv) {
  switch (lv) {
    case level::trace:    return "TRACE";
    case level::debug:    return "DEBUG";
    case level::info:     return "INFO";
    case level::warn:     return "WARN";
    case level::error:    return "ERROR";
    case level::critical: return "CRIT";
  }
  return "?";
}

// ---- Sink management (default: std::cout) -----------------------------------
inline std::ostream* g_sink = &std::cout;
inline std::mutex     g_sink_mtx;                 // used only if no osyncstream
inline std::atomic<level> g_min_level = level::trace;

inline void set_sink(std::ostream& os) { g_sink = &os; }
inline void set_level(level lv)         { g_min_level.store(lv, std::memory_order_relaxed); }

// ---- Time formatting ---------------------------------------------------------
inline void write_now_hms_us(std::ostream& os) {
  using namespace std::chrono;
  const auto tp = system_clock::now();
  const auto t  = system_clock::to_time_t(tp);
  const auto us = duration_cast<microseconds>(tp.time_since_epoch()) % 1'000'000;

  std::tm tm_local{};
#if defined(_WIN32)
  localtime_s(&tm_local, &t);
#else
  localtime_r(&t, &tm_local);
#endif
  os << std::put_time(&tm_local, "%H:%M:%S") << '.'
     << std::setw(6) << std::setfill('0') << us.count();
}

// ---- Core emit (atomic per log call) ----------------------------------------
template <class... Args>
inline void log(level lv,
                std::string_view fmt_prefix, // optional extra prefix
                const src_loc& loc,
                Args&&... args)
{
  if (lv < g_min_level.load(std::memory_order_relaxed)) return;

#if MTLOG_HAS_OSYNCSTREAM
  std::osyncstream out(*g_sink);
  // Prefix
  write_now_hms_us(out);
  out << " [T-" << std::this_thread::get_id() << "] "
      << level_name(lv) << " "
      << "(" << loc.file_name() << ":" << loc.line() << ") ";
  if (!fmt_prefix.empty()) out << " " << fmt_prefix;
  out << " : ";

  // Message (space-separated)
  int i = 0; ((out << (i++ ? " " : "") << std::forward<Args>(args)), ...);
  out << '\n';          // osyncstream flushes atomically on destruction
#else
  // Fallback: build full line then one locked write to sink
  std::ostringstream oss;
  write_now_hms_us(oss);
  oss << " [T-" << std::this_thread::get_id() << "] "
      << level_name(lv) << " "
      << "(" << loc.file_name() << ":" << loc.line() << ")";
  if (!fmt_prefix.empty()) oss << " " << fmt_prefix;
  oss << " : ";

  int i = 0; ((oss << (i++ ? " " : "") << std::forward<Args>(args)), ...);
  oss << '\n';

  std::lock_guard<std::mutex> lk(g_sink_mtx);
  (*g_sink) << oss.str();
  g_sink->flush();
#endif
}

// ---- Convenience macros ------------------------------------------------------
#define TRACE(...)    ::mtlog::log(::mtlog::level::trace,   {}, src_loc::current(), __VA_ARGS__)
#define DEBUG(...)    ::mtlog::log(::mtlog::level::debug,   {}, src_loc::current(), __VA_ARGS__)
#define INFO(...)     ::mtlog::log(::mtlog::level::info,    {}, src_loc::current(), __VA_ARGS__)
#define WARN(...)     ::mtlog::log(::mtlog::level::warn,    {}, src_loc::current(), __VA_ARGS__)
#define ERROR(...)    ::mtlog::log(::mtlog::level::error,   {}, src_loc::current(), __VA_ARGS__)
#define CRITICAL(...) ::mtlog::log(::mtlog::level::critical,{}, src_loc::current(), __VA_ARGS__)

// Optional: custom prefix (e.g., your subsystem tag)
#define MTLOG_TAG(tag, LV, ...) ::mtlog::log((LV), (tag), src_loc::current(), __VA_ARGS__)


} // namespace mtlog
