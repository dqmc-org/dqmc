#include "lattice/lattice_base.h"
#include <iostream>

namespace Lattice {

void LatticeBase::output_k_points(std::ostream& ostream) const {
  // output k stars list
  auto fmt_info = [](int value) { return std::format("{:>20d}", value); };
  auto fmt_kstars = [](double value) {
    return std::format("{:>20.10f}", value);
  };
  ostream << fmt_info(kStarsNum()) << std::endl;
  // loop for inequivalent momentum points
  for (auto i = 0; i < kStarsNum(); ++i) {
    ostream << fmt_info(i);
    // loop for axes of the reciprocal lattice
    for (auto axis = 0; axis < SpaceDim(); ++axis) {
      ostream << fmt_kstars(Index2Momentum(i, axis));
    }
    ostream << std::endl;
  }
}
}  // namespace Lattice
