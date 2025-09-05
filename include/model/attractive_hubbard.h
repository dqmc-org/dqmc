#pragma once

/**
 *  This header file defines Model::AttractiveHubbard class
 *  for describing the attractive fermion hubbard model, which is derived from
 * Model::ModelBase.
 */

#include "model/model_base.h"

// forward declaration
namespace DQMC {
class IO;
}

namespace Model {

// --------------------------------- Derived class Model::AttractiveHubbard
// -------------------------------------
class AttractiveHubbard : public ModelBase {
 protected:
  using RealScalar = double;
  using SpaceTimeMat = Eigen::MatrixXd;

  // Model parameters
  // The Hamiltonian of attractive hubbard model
  //      H  =  -  t   \sum_{<ij>,\sigma} ( c^\dagger_j * c_i + h.c. )
  //            - |U|  \sum_i ( n_up_i - 1/2 ) * ( n_dn_i - 1/2 )
  //            + \mu  \sum_{i,\sigma} ( n_i_sigma )
  RealScalar m_hopping_t{};
  RealScalar m_onsite_u{};
  RealScalar m_chemical_potential{};

  // helping parameter for construction of V matrices
  RealScalar m_alpha{};

  // auxiliary bonsonic fields
  SpaceTimeMat m_bosonic_field{};

  // cached exponentials
  double m_exp_val_avg;
  double m_exp_val_half_diff;

  // pre-allocated buffers to avoid temporary allocations in multiplication functions
  mutable Eigen::VectorXd m_exp_V_col_buffer;
  mutable Eigen::RowVectorXd m_exp_V_row_buffer;

 public:
  explicit AttractiveHubbard(RealScalar hopping_t, RealScalar onsite_u,
                             RealScalar chemical_potential);

  AttractiveHubbard() = delete;
  AttractiveHubbard(const AttractiveHubbard&) = delete;
  AttractiveHubbard& operator=(const AttractiveHubbard&) = delete;
  AttractiveHubbard(AttractiveHubbard&&) = delete;
  AttractiveHubbard& operator=(AttractiveHubbard&&) = delete;

  // ----------------------------------------- Friend class
  // --------------------------------------------- friend class
  // DQMC::IO for reading the bosonic fields from file or
  // outputting the current field configurations to standard output streams
  friend class DQMC::IO;

  // ------------------------------------------ Interfaces
  // ----------------------------------------------

 public:
  RealScalar HoppingT() const override;
  RealScalar OnSiteU() const override;
  RealScalar ChemicalPotential() const override;

  // output model information to stream with consistent formatting
  void output_model_info(std::ostream& ostream) const override;

  // output auxiliary field configuration to stream
  void output_configuration(std::ostream& ostream) const override;

  // read auxiliary field configuration from stream
  void read_auxiliary_field_from_stream(std::istream& infile) override;

  // ----------------------------------- Set up model parameters
  // ----------------------------------------

  void set_model_params(RealScalar hopping_t, RealScalar onsite_u,
                        RealScalar chemical_potential) override;

  // ------------------------------------- Initializations
  // ----------------------------------------------

  virtual void initial(const LatticeBase& lattice, const Walker& walker) override;
  virtual void initial_params(const LatticeBase& lattice, const Walker& walker) override;
  virtual void initial_KV_matrices(const LatticeBase& lattice, const Walker& walker) override;
  void set_bosonic_fields_to_random(std::default_random_engine& rng) override;

  // ------------------------------------- Monte Carlo updates
  // ------------------------------------------

  void update_bosonic_field(TimeIndex time_index, SpaceIndex space_index) override;
  void update_greens_function(Walker& walker, TimeIndex time_index,
                              SpaceIndex space_index) override;
  double get_update_ratio(const Walker& walker, TimeIndex time_index,
                          SpaceIndex space_index) const override;

  // -------------------------------------- Warpping methods
  // --------------------------------------------

  virtual void mult_B_from_left(GreensFunc& green, TimeIndex time_index, Spin spin) const override;
  virtual void mult_B_from_right(GreensFunc& green, TimeIndex time_index, Spin spin) const override;
  virtual void mult_invB_from_left(GreensFunc& green, TimeIndex time_index,
                                   Spin spin) const override;
  virtual void mult_invB_from_right(GreensFunc& green, TimeIndex time_index,
                                    Spin spin) const override;
  virtual void mult_transB_from_left(GreensFunc& green, TimeIndex time_index,
                                     Spin spin) const override;
};
}  // namespace Model
