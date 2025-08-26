#include "utils/assert.h"

#include <cstdio>

namespace Utils {

void report_failure(const char* file_name, int line, const char* function_name,
                    const char* message) {
  std::fprintf(stderr, "%s:%d: DQMC Assertion failed in %s : %s\n", file_name, line, function_name,
               message);
  std::fflush(stderr);
}

}  // namespace Utils
