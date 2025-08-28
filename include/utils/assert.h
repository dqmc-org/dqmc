#pragma once

#include <format>

namespace Utils {

void report_failure(const char* file_name, int line, const char* function_name,
                    const char* message);

}

#ifdef NDEBUG
#define DQMC_ASSERT(condition) ((void)0)
#else
#define DQMC_ASSERT(condition)                                         \
  do {                                                                 \
    if (!(condition)) {                                                \
      Utils::report_failure(__FILE__, __LINE__, __func__, #condition); \
      __builtin_trap();                                                \
    }                                                                  \
  } while (0)
#endif

template <typename... Args>
std::string dqmc_format_error_impl(const char* func, std::string_view format_str, Args&&... args) {
  return std::vformat(std::string("{}: ") + std::string(format_str),
                      std::make_format_args(func, std::forward<Args>(args)...));
}

#if defined(__GNUC__) || defined(__clang__)
#define dqmc_format_error(format_str, ...) \
  dqmc_format_error_impl(__PRETTY_FUNCTION__, format_str __VA_OPT__(, ) __VA_ARGS__)
#else
#define dqmc_format_error(format_str, ...) \
  dqmc_format_error_impl(__func__, format_str __VA_OPT__(, ) __VA_ARGS__)
#endif
