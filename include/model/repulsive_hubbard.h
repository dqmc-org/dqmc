#pragma once

/**
 *  This header file defines Model::RepulsiveHubbard class
 *  for describing the repulsive fermion hubbard model, which is derived from
 * Model::ModelBase.
 */

#include "model/model_base.h"

// forward declaration
namespace DQMC {
class IO;
}

namespace Model {

// --------------------------------- Derived class Model::RepulsiveHubbard
// -------------------------------------
class RepulsiveHubbard : public ModelBase {
 protected:
  using RealScalar = double;
  using SpaceTimeMat = Eigen::MatrixXd;

  // Model parameters
  // The Hamiltonian of repulsive hubbard model
  //      H  =  -  t   \sum_{<ij>,\sigma} ( c^\dagger_j * c_i + h.c. )
  //            + |U|  \sum_i ( n_up_i - 1/2 ) * ( n_dn_i - 1/2 )
  //            + \mu  \sum_{i,\sigma} ( n_i_sigma )
  RealScalar m_hopping_t{};
  RealScalar m_onsite_u{};
  RealScalar m_chemical_potential{};

  // helping parameter for construction of V matrices
  RealScalar m_alpha{};

  // auxiliary bonsonic fields
  SpaceTimeMat m_bosonic_field{};

 public:
  // ----------------------------------------- Friend class
  // --------------------------------------------- friend class
  // DQMC::IO for reading the bosonic fields from file or
  // outputting the current field configurations to standard output streams
  friend class DQMC::IO;

  // ------------------------------------------ Interfaces
  // ----------------------------------------------

  const RealScalar HoppingT() const;
  const RealScalar OnSiteU() const;
  const RealScalar ChemicalPotential() const;

  // output model information to stream with consistent formatting
  void output_model_info(
      std::ostream& ostream,
      const std::function<std::string(const std::string&, const std::string&,
                                      double)>& fmt_param_double,
      const std::string& joiner) const override;

  // ----------------------------------- Set up model parameters
  // ----------------------------------------

  void set_model_params(RealScalar hopping_t, RealScalar onsite_u,
                        RealScalar chemical_potential);

  // ------------------------------------- Initializations
  // ----------------------------------------------

  virtual void initial(const LatticeBase& lattice, const Walker& walker);
  virtual void initial_params(const LatticeBase& lattice, const Walker& walker);
  virtual void initial_KV_matrices(const LatticeBase& lattice,
                                   const Walker& walker);
  void set_bosonic_fields_to_random(std::default_random_engine& rng);

  // ------------------------------------- Monte Carlo updates
  // ------------------------------------------

  void update_bosonic_field(TimeIndex time_index, SpaceIndex space_index);
  void update_greens_function(Walker& walker, TimeIndex time_index,
                              SpaceIndex space_index);
  const double get_update_ratio(Walker& walker, TimeIndex time_index,
                                SpaceIndex space_index) const;

  // -------------------------------------- Warpping methods
  // --------------------------------------------

  virtual void mult_B_from_left(GreensFunc& green, TimeIndex time_index,
                                Spin spin) const;
  virtual void mult_B_from_right(GreensFunc& green, TimeIndex time_index,
                                 Spin spin) const;
  virtual void mult_invB_from_left(GreensFunc& green, TimeIndex time_index,
                                   Spin spin) const;
  virtual void mult_invB_from_right(GreensFunc& green, TimeIndex time_index,
                                    Spin spin) const;
  virtual void mult_transB_from_left(GreensFunc& green, TimeIndex time_index,
                                     Spin spin) const;
};
}  // namespace Model
