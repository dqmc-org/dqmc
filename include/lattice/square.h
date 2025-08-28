#pragma once

/**
 *  This header file defines the Lattice::Square class for 2d square lattice,
 *  derived from the base class Lattice::LatticeBase.
 */

#include <array>
#include <unordered_map>

#include "lattice/lattice_base.h"

namespace Lattice {

// ------------------------ Derived class Lattice::Square for 2d square lattice
// ----------------------------
class Square : public LatticeBase {
 private:
  // some high symmetry points in the reciprocal lattice
  int m_gamma_point_index{};           // (0,0)
  int m_x_point_index{};               // (pi,0)
  int m_m_point_index{};               // (pi,pi)
  LatticeIntVec m_delta_line_index{};  // (0,0)  ->  (pi,0)
  LatticeIntVec m_z_line_index{};      // (pi,0) ->  (pi,pi)
  LatticeIntVec m_sigma_line_index{};  // (0,0)  ->  (pi,pi)

  // defined loop (0,0) -> (pi,0) -> (pi,pi) -> (0,0)
  LatticeIntVec m_gamma2x2m2gamma_loop_index{};

 public:
  explicit Square(const LatticeIntVec& lattice_size);

  Square() = delete;
  Square(const Square&) = delete;
  Square& operator=(const Square&) = delete;
  Square(Square&&) = delete;
  Square& operator=(Square&&) = delete;

  void initial() override;

 private:
  void set_lattice_params(const LatticeIntVec& side_length_vec) override;

 public:
  // Output lattice information
  void output_lattice_info(std::ostream& ostream, int momentum_index) const override;

 private:
  // private initialization functions
  void initial_index2site_table() override;
  void initial_index2momentum_table() override;

  void initial_nearest_neighbour_table() override;
  void initial_displacement_table() override;
  void initial_symmetry_points() override;
  void initial_fourier_factor_table() override;

  void initial_hopping_matrix() override;

  int site_to_index(int x, int y) const;
  std::array<int, 2> index_to_site(int index) const;
};
}  // namespace Lattice
