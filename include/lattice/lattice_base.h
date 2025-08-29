#pragma once

/**
 *  This header file defines the pure virtual abstract class
 * Lattice::LatticeBase for describing the space-discreted lattice where the
 * quantum systems live.
 */

#include <Eigen/Core>
#include <format>
#include <iostream>
#include <unordered_map>

#include "utils/assert.h"

namespace Lattice {

// -------------------------- Pure virtual base class Lattice::LatticeBase
// ----------------------------
class LatticeBase {
 protected:
  int m_space_dim{};            // dimension of the space
  int m_side_length{};          // side length of the lattice
  int m_space_size{};           // total number of lattice sites
  int m_coordination_number{};  // coordination number of the lattice
  int m_num_k_stars{};          // number of k stars ( inequivalent momentum points )

  // hopping matrix, depending only on the topology of lattice
  // hopping constants are normalized to 1.0 .
  Eigen::MatrixXd m_hopping_matrix{};

  // Matrix structure for storing the nearest neighbours of each lattice site
  // the matrix shape is SpaceSize * Coordination number
  Eigen::MatrixXi m_nearest_neighbour_table{};
  // todo: next nearest neighbours
  // Matrixint m_next_nearest_neighbour_table{};

  // Matrix structure for storing the map from site index to the site vector
  // with the shape of SpaceSize * SpaceDim
  Eigen::MatrixXi m_index2site_table{};

  // table of the displacement between any two sites i and j, pointing from i to
  // j the displacement is represented by a site index due to the periodic
  // boundary condition, and the shape of the table is SpaceSize * SpaceSize.
  Eigen::MatrixXi m_displacement_table{};

  // the map from momentum index to the lattice momentum in the reciprocal
  // lattice the number of rows should be equal to the number of inequivalent
  // momentum points (k stars), and the columns is the space dimension.
  Eigen::MatrixXd m_index2momentum_table{};

  // table of fourier transformation factor
  // e.g. Re( exp(ikx) ) for lattice site x and momentum k
  Eigen::MatrixXd m_fourier_factor_table{};

  // all inequivalent momentum points ( k stars ) in the reciprocal lattice
  std::vector<int> m_k_stars_index{};

  // Momentum points and lines
  std::unordered_map<std::string, int> m_momentum_points;
  std::unordered_map<std::string, std::vector<int>> m_momentum_lists;

 public:
  LatticeBase() = default;
  virtual ~LatticeBase() = default;

  // ---------------------------- Set up lattice parameters
  // ------------------------------ read lattice params from a vector of side
  // lengths
  virtual void set_lattice_params(const std::vector<int>& side_length_vec) = 0;

  // --------------------------------- interfaces
  // ----------------------------------------

  int space_dim() const { return this->m_space_dim; }

  int space_size() const { return this->m_space_size; }

  int side_length() const { return this->m_side_length; }

  int coordination_number() const { return this->m_coordination_number; }

  int k_stars_num() const { return this->m_num_k_stars; }

  const std::vector<int>& k_stars_index() const { return this->m_k_stars_index; }

  const Eigen::MatrixXd& hopping_matrix() const { return this->m_hopping_matrix; }

  const Eigen::MatrixXd& fourier_factor() const { return this->m_fourier_factor_table; }

  int displacement(int site1_index, int site2_index) const {
    DQMC_ASSERT(site1_index >= 0 && site1_index < this->m_space_size);
    DQMC_ASSERT(site2_index >= 0 && site2_index < this->m_space_size);
    return this->m_displacement_table(site1_index, site2_index);
  }

  double fourier_factor(int site_index, int momentum_index) const {
    DQMC_ASSERT(site_index >= 0 && site_index < this->m_space_size);
    DQMC_ASSERT(momentum_index >= 0 && momentum_index < this->m_num_k_stars);
    return this->m_fourier_factor_table(site_index, momentum_index);
  }

  int nearest_neighbour(int site_index, int direction) const {
    DQMC_ASSERT(site_index >= 0 && site_index < this->m_space_size);
    DQMC_ASSERT(direction >= 0 && direction < this->m_coordination_number);
    return this->m_nearest_neighbour_table(site_index, direction);
  }

  Eigen::MatrixXi::ConstRowXpr get_neighbors(int site_index) const {
    DQMC_ASSERT(site_index >= 0 && site_index < this->m_space_size);
    return this->m_nearest_neighbour_table.row(site_index);
  }

  Eigen::MatrixXi::ConstRowXpr index_to_site(int site_index) const {
    DQMC_ASSERT(site_index >= 0 && site_index < this->m_space_size);
    return this->m_index2site_table.row(site_index);
  }

  int index_to_site(int site_index, int axis) const {
    DQMC_ASSERT(site_index >= 0 && site_index < this->m_space_size);
    DQMC_ASSERT(axis >= 0 && axis < this->m_space_dim);
    return this->m_index2site_table(site_index, axis);
  }

  Eigen::MatrixXd::ConstRowXpr index_to_momentum(int momentum_index) const {
    DQMC_ASSERT(momentum_index >= 0 && momentum_index < this->m_num_k_stars);
    return this->m_index2momentum_table.row(momentum_index);
  }

  double index_to_momentum(int momentum_index, int axis) const {
    DQMC_ASSERT(momentum_index >= 0 && momentum_index < this->m_num_k_stars);
    DQMC_ASSERT(axis >= 0 && axis < this->m_space_dim);
    return this->m_index2momentum_table(momentum_index, axis);
  }

  // -------------------------------- Initializations
  // ------------------------------------

  virtual void initial() = 0;
  virtual void initial_hopping_matrix() = 0;
  virtual void initial_index2site_table() = 0;
  virtual void initial_nearest_neighbour_table() = 0;
  virtual void initial_displacement_table() = 0;
  virtual void initial_index2momentum_table() = 0;
  virtual void initial_symmetry_points() = 0;
  virtual void initial_fourier_factor_table() = 0;

  // Output lattice information - self-documenting interface
  virtual void output_lattice_info(std::ostream& ostream, int momentum_index) const = 0;

  // Output list of inequivalent momentum points (k stars)
  void output_k_points(std::ostream& ostream) const;

  // Momentum hashmaps
  const std::unordered_map<std::string, int>& momentum_points() const { return m_momentum_points; };

  const std::unordered_map<std::string, std::vector<int>>& momentum_lists() const {
    return m_momentum_lists;
  };
};

}  // namespace Lattice
