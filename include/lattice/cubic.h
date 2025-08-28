#pragma once

/**
 *  This header file defines the Lattice::Cubic class for 3d cubic lattice,
 *  derived from the base class Lattice::LatticeBase.
 */

#include <array>

#include "lattice/lattice_base.h"

namespace Lattice {

// ------------------------ Derived class Lattice::Cubic for 3d cubic lattice
// ----------------------------
class Cubic : public LatticeBase {
 private:
  // some high symmetry points in the reciprocal lattice
  LatticeInt m_gamma_point_index{};     // (0,0,0)
  LatticeInt m_x_point_index{};         // (pi,0,0)
  LatticeInt m_m_point_index{};         // (pi,pi,0)
  LatticeInt m_r_point_index{};         // (pi,pi,pi)
  LatticeIntVec m_delta_line_index{};   // (0,0,0)   ->  (pi,0,0)
  LatticeIntVec m_z_line_index{};       // (pi,0,0)  ->  (pi,pi,0)
  LatticeIntVec m_sigma_line_index{};   // (0,0,0)   ->  (pi,pi,0)
  LatticeIntVec m_lambda_line_index{};  // (0,0,0)   ->  (pi,pi,pi)
  LatticeIntVec m_s_line_index{};       // (pi,0,0)  ->  (pi,pi,pi)
  LatticeIntVec m_t_line_index{};       // (pi,pi,0) ->  (pi,pi,pi)

 public:
  explicit Cubic(const LatticeIntVec& lattice_size);

  Cubic() = delete;
  Cubic(const Cubic&) = delete;
  Cubic& operator=(const Cubic&) = delete;
  Cubic(Cubic&&) = delete;
  Cubic& operator=(Cubic&&) = delete;

  void initial() override;

 private:
  void set_lattice_params(const LatticeIntVec& side_length_vec) override;

 public:
  // interfaces for high symmetry momentum points
  LatticeInt GammaPointIndex() const;
  LatticeInt XPointIndex() const;
  LatticeInt MPointIndex() const;
  LatticeInt RPointIndex() const;
  const LatticeIntVec& DeltaLineIndex() const;
  const LatticeIntVec& ZLineIndex() const;
  const LatticeIntVec& SigmaLineIndex() const;
  const LatticeIntVec& LambdaLineIndex() const;
  const LatticeIntVec& SLineIndex() const;
  const LatticeIntVec& TLineIndex() const;

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
