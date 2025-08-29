#pragma once

/**
 *  This header file defines the Lattice::Chain class for a 1D chain lattice,
 *  derived from the base class Lattice::LatticeBase.
 */

#include <array>

#include "lattice/lattice_base.h"

namespace Lattice {

// ------------------------ Derived class Lattice::Chain for 1d chain lattice
// ----------------------------
class Chain : public LatticeBase {
 private:
  // some high symmetry points in the reciprocal lattice
  int m_gamma_point_index{};              // k = 0
  int m_x_point_index{};                  // k = pi
  std::vector<int> m_delta_line_index{};  // 0 -> pi

 public:
  explicit Chain(const std::vector<int>& lattice_size);

  Chain() = delete;
  Chain(const Chain&) = delete;
  Chain& operator=(const Chain&) = delete;
  Chain(Chain&&) = delete;
  Chain& operator=(Chain&&) = delete;

  void initial() override;

 private:
  void set_lattice_params(const std::vector<int>& side_length_vec) override;

 public:
  // Output lattice information
  void output_lattice_info(std::ostream& ostream, int momentum_index) const override;

 private:
  // private initialization functions
  void initial_index_to_site_table() override;
  void initial_index_to_momentum_table() override;

  void initial_nearest_neighbor_table() override;
  void initial_displacement_table() override;
  void initial_symmetry_points() override;
  void initial_fourier_factor_table() override;

  void initial_hopping_matrix() override;

  int site_to_index(int x) const;
  int index_to_site(int index) const;
};
}  // namespace Lattice