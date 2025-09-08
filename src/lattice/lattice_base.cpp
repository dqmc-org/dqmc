#include "lattice/lattice_base.h"

#include <iostream>

namespace Lattice {

void LatticeBase::output_k_points(std::ostream& ostream) const {
  // output k stars list
  std::string header = "index";
  const std::vector<std::string> axes = {"kx", "ky", "kz"};
  for (int i = 0; i < space_dim(); ++i) {
    header += "," + axes[i];
  }
  ostream << header << "\n";

  // loop for inequivalent momentum points
  for (auto i = 0; i < k_stars_num(); ++i) {
    ostream << i;
    // loop for axes of the reciprocal lattice
    for (auto axis = 0; axis < space_dim(); ++axis) {
      ostream << "," << index_to_momentum(i, axis);
    }
    ostream << "\n";
  }
}
}  // namespace Lattice
