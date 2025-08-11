#ifndef LATTICE_INDEXER_H
#define LATTICE_INDEXER_H

#include <vector>

class Indexer {
 public:
  Indexer() = default;

  Indexer(std::vector<int> dimensions)
      : m_dimensions(std::move(dimensions)),
        m_total_size(compute_total_size()) {}

  [[nodiscard]] int to_orbital(const std::vector<int>& coordinates) const {
    assert(coordinates.size() == m_dimensions.size());
    int orbital = 0;
    int multiplier = 1;
    for (int i = 0; i < m_dimensions.size(); ++i) {
      assert(coordinates[i] < m_dimensions[i]);
      orbital += coordinates[i] * multiplier;
      multiplier *= m_dimensions[i];
    }
    return orbital;
  }

  [[nodiscard]] std::vector<int> from_orbital(int orbital) const {
    assert(orbital < size());
    std::vector<int> coordinates(m_dimensions.size());
    for (int i = 0; i < m_dimensions.size(); ++i) {
      coordinates[i] = orbital % m_dimensions[i];
      orbital /= m_dimensions[i];
    }
    return coordinates;
  }

  const std::vector<int>& dimensions() const { return m_dimensions; }

  int dimension(int i) const {
    assert(i < m_dimensions.size());
    return m_dimensions[i];
  }

  int size() const { return m_total_size; }

 private:
  std::vector<int> m_dimensions;
  int m_total_size{};

  int compute_total_size() {
    m_total_size = 1;
    for (int dim : m_dimensions) {
      m_total_size *= dim;
    }
    return m_total_size;
  }
};
#endif  // LATTICE_INDEXER_H
