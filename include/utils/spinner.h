#pragma once

#include <iostream>

namespace Utils {
class Spinner {
 public:
  Spinner() : m_index(0) {}

  void spin() {
    if (m_index % 100 == 0) {
      std::cout << '.' << std::flush;
    } else if (m_index > 50 * 100) {
      std::cout << '\n' << std::flush;
      m_index = 0;
    }
    m_index++;
  }

  ~Spinner() { std::cout << '\r' << std::flush; }

 private:
  int m_index;
};
}  // namespace Utils
