#pragma once

#include <format>
#include <string>

namespace Utils {
namespace FormatOutput {

static constexpr int DESC_WIDTH = 30;
static constexpr int ARROW_WIDTH = 7;
static constexpr int VALUE_WIDTH = 24;

template <typename T>
inline std::string display(const std::string& desc, T value) {
  if constexpr (std::floating_point<T>) {
    return std::format("{:>{}}{:>{}}{:>{}.3f}\n", desc, DESC_WIDTH, "->", ARROW_WIDTH, value,
                       VALUE_WIDTH);
  } else {
    return std::format("{:>{}}{:>{}}{:>{}}\n", desc, DESC_WIDTH, "->", ARROW_WIDTH, value,
                       VALUE_WIDTH);
  }
}

}  // namespace FormatOutput
}  // namespace Utils
