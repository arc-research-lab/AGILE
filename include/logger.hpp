#pragma once


#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <mutex>

#if defined(_WIN32)
  #include <io.h>
  #define isatty _isatty
  #define fileno _fileno
#else
  #include <unistd.h>
#endif

namespace host_logger {

enum class Level : int { Trace = 0, Debug, Info, Warn, Error, Fatal };

inline const char* level_to_string(Level lvl) {
  switch (lvl) {
    case Level::Trace: return "TRACE";
    case Level::Debug: return "DEBUG";
    case Level::Info:  return "INFO";
    case Level::Warn:  return "WARN";
    case Level::Error: return "ERROR";
    case Level::Fatal: return "FATAL";
    default:           return "UNKNOWN";
  }
}

/* ANSI colors */
inline const char* level_color(Level lvl) {
  switch (lvl) {
    case Level::Trace: return "\033[90m";   // bright black / gray
    case Level::Debug: return "\033[36m";   // cyan
    case Level::Info:  return "\033[32m";   // green
    case Level::Warn:  return "\033[33m";   // yellow
    case Level::Error: return "\033[31m";   // red
    case Level::Fatal: return "\033[1;31m"; // bold red
    default:           return "";
  }
}

inline constexpr const char* color_reset() { return "\033[0m"; }

// ---- configuration ----

inline Level& min_level() {
  static Level lvl = Level::Trace;
  return lvl;
}

inline FILE*& sink() {
  static FILE* s = stdout;
  return s;
}

inline bool& use_color() {
  static bool enabled = true;
  return enabled;
}

inline std::mutex& mu() {
  static std::mutex m;
  return m;
}

inline void set_min_level(Level lvl) { min_level() = lvl; }
inline void set_sink(FILE* f) { sink() = (f ? f : stdout); }
inline void enable_color(bool v) { use_color() = v; }

inline bool is_tty() {
  return isatty(fileno(sink()));
}

// ---- core logging ----

inline void vlogf(Level lvl,
                  const char* tag,
                  const char* fmt,
                  va_list ap) {
  if (static_cast<int>(lvl) < static_cast<int>(min_level())) return;

  std::lock_guard<std::mutex> lock(mu());

  const bool color = use_color() && is_tty();

  if (color) std::fputs(level_color(lvl), sink());

  std::fprintf(sink(),
               "[%s][%s]\t ",
               level_to_string(lvl),
               (tag && *tag) ? tag : "GLOBAL");

  if (color) std::fputs(color_reset(), sink());

  std::vfprintf(sink(), fmt, ap);

  // auto newline if missing
  const char* p = fmt;
  while (*p) ++p;
  if (p == fmt || *(p - 1) != '\n') std::fputc('\n', sink());

  std::fflush(sink());

  if (lvl == Level::Fatal) std::abort();
}

inline void logf(Level lvl,
                 const char* tag,
                 const char* fmt,
                 ...) {
  va_list ap;
  va_start(ap, fmt);
  vlogf(lvl, tag, fmt, ap);
  va_end(ap);
}

inline void rawf(const char* fmt, ...) {
  std::lock_guard<std::mutex> lock(mu());

  va_list ap;
  va_start(ap, fmt);
  std::vfprintf(sink(), fmt, ap);
  va_end(ap);

  std::fflush(sink());
}

} // namespace host_logger

// ---- macros ----
// Usage: LOG_INFO(TAG_STORAGE, "msg: %s", xxx);

#define LOG_TRACE(tag, ...) ::host_logger::logf(::host_logger::Level::Trace, tag, __VA_ARGS__)
#define LOG_DEBUG(tag, ...) ::host_logger::logf(::host_logger::Level::Debug, tag, __VA_ARGS__)
#define LOG_INFO(tag, ...)  ::host_logger::logf(::host_logger::Level::Info,  tag, __VA_ARGS__)
#define LOG_WARN(tag, ...)  ::host_logger::logf(::host_logger::Level::Warn,  tag, __VA_ARGS__)
#define LOG_ERROR(tag, ...) ::host_logger::logf(::host_logger::Level::Error, tag, __VA_ARGS__)
#define LOG_FATAL(tag, ...) ::host_logger::logf(::host_logger::Level::Fatal, tag, __VA_ARGS__)


// ---- optional tag definitions (examples) ----
// Put these in a common header if you want shared tags.
/*
inline constexpr const char* TAG_STORAGE = "STORAGE";
inline constexpr const char* TAG_IPC     = "IPC";
inline constexpr const char* TAG_F2FS    = "F2FS";
*/