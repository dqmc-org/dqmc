// Copyright (c) 2016 Prakhar Srivastav <prakhar@prakhar.me>
// Copyright (c) 2025 Matheus de Sousa <msousa3145@gmail.com>

#pragma once

#include <chrono>
#include <format>
#include <iostream>
#include <string>

namespace DQMC {
class ProgressBar {
 private:
  std::size_t ticks = 0;
  std::size_t total_ticks;
  std::size_t bar_width;
  char complete_char = '=';
  char incomplete_char = ' ';
  std::chrono::steady_clock::time_point start_time =
      std::chrono::steady_clock::now();

 public:
  ProgressBar(std::size_t total, std::size_t width, char complete,
              char incomplete)
      : total_ticks{total},
        bar_width{width},
        complete_char{complete},
        incomplete_char{incomplete} {}

  ProgressBar(std::size_t total, std::size_t width)
      : total_ticks{total}, bar_width{width} {}

  std::size_t operator++() { return ++ticks; }

  void display() const {
    float progress = static_cast<float>(ticks) / total_ticks;
    int pos = static_cast<int>(bar_width * progress);
    std::chrono::steady_clock::time_point now =
        std::chrono::steady_clock::now();
    std::chrono::duration<double> time_elapsed = now - start_time;

    std::string bar_str;
    bar_str.reserve(bar_width + 3);

    bar_str += '[';
    for (int i = 0; i < static_cast<int>(bar_width); ++i) {
      if (i < pos) {
        bar_str += complete_char;
      } else if (i == pos) {
        bar_str += '>';
      } else {
        bar_str += incomplete_char;
      }
    }
    bar_str += ']';

    std::cout << std::format("{} {:>3}% {:.3f}s\r", bar_str,
                             static_cast<int>(progress * 100.0),
                             time_elapsed.count());
    std::cout.flush();
  }

  void done() const {
    display();
    std::cout << '\n';
  }
};
}  // namespace DQMC
