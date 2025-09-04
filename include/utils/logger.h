#pragma once

#include <format>
#include <fstream>
#include <memory>
#include <string>

namespace Utils {
class Logger {
 public:
  enum class LogLevel { DEBUG, INFO, WARNING, ERROR };

  Logger(const Logger&) = delete;
  Logger& operator=(const Logger&) = delete;
  Logger(Logger&&) = delete;
  Logger& operator=(Logger&&) = delete;

  static Logger& the() {
    static Logger instance;
    return instance;
  }

  void set_level(LogLevel level);
  void set_to_console(bool enable);
  bool set_file(const std::string& filename);
  void log(LogLevel level, const std::string& message);
  void debug(const std::string& message);
  void info(const std::string& message);
  void warn(const std::string& message);
  void error(const std::string& message);

  template <typename... Args>
  void log_fmt(LogLevel level, std::format_string<Args...> fmt, Args&&... args) {
    std::string formatted_message = std::format(fmt, std::forward<Args>(args)...);
    log(level, formatted_message);
  }

  template <typename... Args>
  void debug(std::format_string<Args...> fmt, Args&&... args) {
    log_fmt(LogLevel::DEBUG, fmt, std::forward<Args>(args)...);
  }
  template <typename... Args>
  void info(std::format_string<Args...> fmt, Args&&... args) {
    log_fmt(LogLevel::INFO, fmt, std::forward<Args>(args)...);
  }
  template <typename... Args>
  void warn(std::format_string<Args...> fmt, Args&&... args) {
    log_fmt(LogLevel::WARNING, fmt, std::forward<Args>(args)...);
  }
  template <typename... Args>
  void error(std::format_string<Args...> fmt, Args&&... args) {
    log_fmt(LogLevel::ERROR, fmt, std::forward<Args>(args)...);
  }

 private:
  Logger();
  ~Logger() = default;

  std::string level_to_string(LogLevel level) const;

  LogLevel m_level;
  bool m_console;
  std::string m_path;
  std::unique_ptr<std::ofstream> m_file;
};
}  // namespace Utils
