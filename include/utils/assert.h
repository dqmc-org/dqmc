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
