#pragma once

/**
 *  This header file defines the pure virtual abstract class
 * Lattice::LatticeBase for describing the space-discreted lattice where the
 * quantum systems live.
 */

#include <Eigen/Core>
#include <format>
#include <iostream>

#include "utils/assert.h"

namespace Lattice {

using LatticeBool = bool;
using LatticeInt = int;
using LatticeDouble = double;
using LatticeIntVec = std::vector<int>;
using MatrixDouble = Eigen::MatrixXd;
using VectorDouble = Eigen::VectorXd;
using MatrixInt = Eigen::MatrixXi;
using VectorInt = Eigen::VectorXi;

// -------------------------- Pure virtual base class Lattice::LatticeBase
// ----------------------------
class LatticeBase {
 protected:
  LatticeBool m_initial_status{false};  // status of initialization

  LatticeInt m_space_dim{};            // dimension of the space
  LatticeInt m_side_length{};          // side length of the lattice
  LatticeInt m_space_size{};           // total number of lattice sites
  LatticeInt m_coordination_number{};  // coordination number of the lattice
  LatticeInt
      m_num_k_stars{};  // number of k stars ( inequivalent momentum points )

  // hopping matrix, depending only on the topology of lattice
  // hopping constants are normalized to 1.0 .
  MatrixDouble m_hopping_matrix{};

  // Matrix structure for storing the nearest neighbours of each lattice site
  // the matrix shape is SpaceSize * Coordination number
  MatrixInt m_nearest_neighbour_table{};
  // todo: next nearest neighbours
  // MatrixInt m_next_nearest_neighbour_table{};

  // Matrix structure for storing the map from site index to the site vector
  // with the shape of SpaceSize * SpaceDim
  MatrixInt m_index2site_table{};

  // table of the displacement between any two sites i and j, pointing from i to
  // j the displacement is represented by a site index due to the periodic
  // boundary condition, and the shape of the table is SpaceSize * SpaceSize.
  MatrixInt m_displacement_table{};

  // the map from momentum index to the lattice momentum in the reciprocal
  // lattice the number of rows should be equal to the number of inequivalent
  // momentum points (k stars), and the columns is the space dimension.
  MatrixDouble m_index2momentum_table{};

  // table of fourier transformation factor
  // e.g. Re( exp(ikx) ) for lattice site x and momentum k
  MatrixDouble m_fourier_factor_table{};

  // all inequivalent momentum points ( k stars ) in the reciprocal lattice
  LatticeIntVec m_k_stars_index{};

 public:
  LatticeBase() = default;
  virtual ~LatticeBase() = default;

  // ---------------------------- Set up lattice parameters
  // ------------------------------ read lattice params from a vector of side
  // lengths
  virtual void set_lattice_params(const LatticeIntVec& side_length_vec) = 0;

  // --------------------------------- Interfaces
  // ----------------------------------------

  LatticeBool InitialStatus() const { return this->m_initial_status; }

  LatticeInt SpaceDim() const { return this->m_space_dim; }

  LatticeInt SpaceSize() const { return this->m_space_size; }

  LatticeInt SideLength() const { return this->m_side_length; }

  LatticeInt CoordinationNumber() const { return this->m_coordination_number; }

  LatticeInt kStarsNum() const { return this->m_num_k_stars; }

  const LatticeIntVec& kStarsIndex() const { return this->m_k_stars_index; }

  const MatrixDouble& HoppingMatrix() const { return this->m_hopping_matrix; }

  const MatrixDouble& FourierFactor() const {
    return this->m_fourier_factor_table;
  }

  LatticeInt Displacement(const LatticeInt site1_index,
                          const LatticeInt site2_index) const {
    DQMC_ASSERT(site1_index >= 0 && site1_index < this->m_space_size);
    DQMC_ASSERT(site2_index >= 0 && site2_index < this->m_space_size);
    return this->m_displacement_table(site1_index, site2_index);
  }

  LatticeDouble FourierFactor(const LatticeInt site_index,
                              const LatticeInt momentum_index) const {
    DQMC_ASSERT(site_index >= 0 && site_index < this->m_space_size);
    DQMC_ASSERT(momentum_index >= 0 && momentum_index < this->m_num_k_stars);
    return this->m_fourier_factor_table(site_index, momentum_index);
  }

  LatticeInt NearestNeighbour(const LatticeInt site_index,
                              const LatticeInt direction) const {
    DQMC_ASSERT(site_index >= 0 && site_index < this->m_space_size);
    DQMC_ASSERT(direction >= 0 && direction < this->m_coordination_number);
    return this->m_nearest_neighbour_table(site_index, direction);
  }

  VectorInt GetNeighbors(const LatticeInt site_index) const {
    DQMC_ASSERT(site_index >= 0 && site_index < this->m_space_size);
    return this->m_nearest_neighbour_table.row(site_index);
  }

  VectorInt Index2Site(const LatticeInt site_index) const {
    DQMC_ASSERT(site_index >= 0 && site_index < this->m_space_size);
    return this->m_index2site_table.row(site_index);
  }

  LatticeInt Index2Site(const LatticeInt site_index,
                        const LatticeInt axis) const {
    DQMC_ASSERT(site_index >= 0 && site_index < this->m_space_size);
    DQMC_ASSERT(axis >= 0 && axis < this->m_space_dim);
    return this->m_index2site_table(site_index, axis);
  }

  VectorDouble Index2Momentum(const LatticeInt momentum_index) const {
    DQMC_ASSERT(momentum_index >= 0 && momentum_index < this->m_num_k_stars);
    return this->m_index2momentum_table.row(momentum_index);
  }

  LatticeDouble Index2Momentum(const LatticeInt momentum_index,
                               const LatticeInt axis) const {
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
  virtual void output_lattice_info(std::ostream& ostream,
                                   int momentum_index) const = 0;

  // Output list of inequivalent momentum points (k stars)
  void output_k_points(std::ostream& ostream) const;
};

}  // namespace Lattice
