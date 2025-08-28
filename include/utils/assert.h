#pragma once

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

#if defined(__GNUC__) || defined(__clang__)
#define dqmc_format_error(format_string, ...) \
  std::format("{}: " format_string, __PRETTY_FUNCTION__, ##__VA_ARGS__)
#else
#define dqmc_format_error(format_string, ...) \
  std::format("{}: " format_string, __func__, ##__VA_ARGS__)
#endif
