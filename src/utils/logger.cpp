#include "utils/logger.h"

#include <unistd.h>

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace Utils {
Logger::Logger() : m_level(LogLevel::Info), m_console(true), m_file(nullptr) {}

void Logger::set_level(LogLevel level) { m_level = level; }

void Logger::set_to_console(bool enable) { m_console = enable; }

bool Logger::set_file(const std::string& filename) {
  m_path = filename;
  auto newFile = std::make_unique<std::ofstream>(m_path, std::ios::app);

  if (newFile && newFile->is_open()) {
    m_file = std::move(newFile);
    return true;
  } else {
    std::cerr << "Error: Could not open log file: " << m_path << std::endl;
    m_file.reset();
    return false;
  }
}

std::string Logger::level_to_string(LogLevel level) const {
  switch (level) {
    case LogLevel::Debug:
      return "DEBUG";
    case LogLevel::Info:
      return "INFO";
    case LogLevel::Warning:
      return "WARN";
    case LogLevel::Error:
      return "ERROR";
    default:
      return "UNKNOWN";
  }
}

void Logger::log(LogLevel level, const std::string& message) {
  if (level < m_level) {
    return;
  }
  std::string entry = "[" + level_to_string(level) + "] " + message;

  if (m_console) {
    std::cout << entry << std::endl;
  }

  if (m_file && m_file->is_open()) {
    *m_file << entry << std::endl;
  } else if (!m_path.empty()) {
    if (m_console) {
      std::cerr << "[ERROR] Failed to write to log file: " << m_path << std::endl;
    } else {
      std::cerr << "[ERROR] Failed to write to log file: " << m_path << std::endl;
      std::cerr << "[ERROR] Original message: " << message << std::endl;
    }
  }
}

void Logger::debug(const std::string& message) { log(LogLevel::Debug, message); }

void Logger::info(const std::string& message) { log(LogLevel::Info, message); }

void Logger::warn(const std::string& message) { log(LogLevel::Warning, message); }

void Logger::error(const std::string& message) { log(LogLevel::Error, message); }
}  // namespace Utils
