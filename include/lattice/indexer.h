#ifndef LATTICE_INDEXER_H
#define LATTICE_INDEXER_H

#include <stdexcept>
#include <vector>

class Indexer {
 public:
  Indexer() = default;

  Indexer(std::vector<int> dimensions) : m_dimensions(std::move(dimensions)) {}

  [[nodiscard]] int to_orbital(const std::vector<int>& coordinates) const {
    if (coordinates.size() != m_dimensions.size()) {
      throw std::out_of_range("Invalid number of coordinates");
    }

    int orbital = 0;
    int multiplier = 1;
    for (int i = 0; i < m_dimensions.size(); ++i) {
      if (coordinates[i] >= m_dimensions[i]) {
        throw std::out_of_range("Coordinates out of bounds");
      }
      orbital += coordinates[i] * multiplier;
      multiplier *= m_dimensions[i];
    }
    return orbital;
  }

  [[nodiscard]] std::vector<int> from_orbital(int orbital) const {
    if (orbital >= total_size()) {
      throw std::out_of_range("Orbital index out of bounds");
    }

    std::vector<int> coordinates(m_dimensions.size());
    for (int i = 0; i < m_dimensions.size(); ++i) {
      coordinates[i] = orbital % m_dimensions[i];
      orbital /= m_dimensions[i];
    }
    return coordinates;
  }

  const std::vector<int>& dimensions() const { return m_dimensions; }

  int dimension(int i) const {
    if (i >= m_dimensions.size()) {
      throw std::out_of_range("Dimension is out of bounds");
    }
    return m_dimensions[i];
  }

  int size() const { return total_size(); }

 private:
  std::vector<int> m_dimensions;

  int total_size() const {
    int size = 1;
    for (int dim : m_dimensions) {
      size *= dim;
    }
    return size;
  }
};
#endif  // LATTICE_INDEXER_H
