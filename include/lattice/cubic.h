#pragma once

/**
 *  This header file defines the Lattice::Cubic class for 3d cubic lattice,
 *  derived from the base class Lattice::LatticeBase.
 */

#include <array>
#include <unordered_map>

#include "lattice/lattice_base.h"

namespace Lattice {

// ------------------------ Derived class Lattice::Cubic for 3d cubic lattice
// ----------------------------
class Cubic : public LatticeBase {
 private:
  // some high symmetry points in the reciprocal lattice
  int m_gamma_point_index{};               // (0,0,0)
  int m_x_point_index{};                   // (pi,0,0)
  int m_m_point_index{};                   // (pi,pi,0)
  int m_r_point_index{};                   // (pi,pi,pi)
  std::vector<int> m_delta_line_index{};   // (0,0,0)   ->  (pi,0,0)
  std::vector<int> m_z_line_index{};       // (pi,0,0)  ->  (pi,pi,0)
  std::vector<int> m_sigma_line_index{};   // (0,0,0)   ->  (pi,pi,0)
  std::vector<int> m_lambda_line_index{};  // (0,0,0)   ->  (pi,pi,pi)
  std::vector<int> m_s_line_index{};       // (pi,0,0)  ->  (pi,pi,pi)
  std::vector<int> m_t_line_index{};       // (pi,pi,0) ->  (pi,pi,pi)

  std::unordered_map<std::string, int> m_momentum_points;
  std::unordered_map<std::string, std::vector<int>> m_momentum_lists;

 public:
  explicit Cubic(const std::vector<int>& lattice_size);

  Cubic() = delete;
  Cubic(const Cubic&) = delete;
  Cubic& operator=(const Cubic&) = delete;
  Cubic(Cubic&&) = delete;
  Cubic& operator=(Cubic&&) = delete;

  void initial() override;

 private:
  void set_lattice_params(const std::vector<int>& side_length_vec) override;

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

  int site_to_index(int x, int y, int z) const;
  std::array<int, 3> index_to_site(int index) const;
};
}  // namespace Lattice
