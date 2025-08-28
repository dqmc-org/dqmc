#include "lattice/lattice_base.h"

#include <iostream>

namespace Lattice {

void LatticeBase::output_k_points(std::ostream& ostream) const {
  // output k stars list
  auto fmt_info = [](int value) { return std::format("{:>20d}", value); };
  auto fmt_kstars = [](double value) { return std::format("{:>20.10f}", value); };
  ostream << fmt_info(k_stars_num()) << std::endl;
  // loop for inequivalent momentum points
  for (auto i = 0; i < k_stars_num(); ++i) {
    ostream << fmt_info(i);
    // loop for axes of the reciprocal lattice
    for (auto axis = 0; axis < space_dim(); ++axis) {
      ostream << fmt_kstars(index_to_momentum(i, axis));
    }
    ostream << std::endl;
  }
}
}  // namespace Lattice
