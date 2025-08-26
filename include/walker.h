#pragma once

/**
 *  This header file defines the crucial class DQMC::Walker
 *  to organize the entire dqmc program.
 */

#include <Eigen/Core>
#include <memory>
#include <random>
#include <vector>

#include "svd_stack.h"

namespace Model {
class ModelBase;
}
namespace Lattice {
class LatticeBase;
}
namespace Measure {
class MeasureHandler;
}

namespace DQMC {

// forward declaration
class Initializer;

using LatticeBase = Lattice::LatticeBase;
using ModelBase = Model::ModelBase;
using MeasureHandler = Measure::MeasureHandler;

// ---------------------------- Crucial class DQMC::Walker
// ----------------------------
class Walker {
 private:
  using TimeIndex = int;
  using RealScalar = double;
  using RealScalarVec = Eigen::VectorXd;
  using SvdStack = Utils::SvdStack;

  using GreensFunc = Eigen::MatrixXd;
  using GreensFuncVec = std::vector<Eigen::MatrixXd>;

  // --------------------------------- Walker params
  // ---------------------------------------------

  int m_space_size{};            // number of space sites
  int m_time_size{};             // number of time slices
  RealScalar m_beta{};           // inverse temperature beta
  RealScalar m_time_interval{};  // interval of imaginary-time grids
  int m_current_time_slice{};    // helping params to record current time slice

  // ----------------------- ( Equal-time and dynamic ) Greens functions
  // -------------------------

  // Equal-time green's functions, which is the most crucial quantities during
  // dqmc simulations for spin-1/2 systems, we label the spin index with up and
  // down
  GreensFunc m_green_tt_up{}, m_green_tt_dn{};
  GreensFuncVec m_vec_green_tt_up{}, m_vec_green_tt_dn{};

  // Time-displaced green's functions G(t,0) and G(0,t)
  // important for time-displaced measurements of physical observables
  GreensFunc m_green_t0_up{}, m_green_t0_dn{};
  GreensFunc m_green_0t_up{}, m_green_0t_dn{};
  GreensFuncVec m_vec_green_t0_up{}, m_vec_green_t0_dn{};
  GreensFuncVec m_vec_green_0t_up{}, m_vec_green_0t_dn{};

  bool m_is_equaltime{};
  bool m_is_dynamic{};

  // ------------------------- SvdStack for numerical stabilization
  // ------------------------------

  // Utils::SvdStack class
  // for efficient svd decompositions and numerical stabilization
  SvdStack m_svd_stack_left_up{};
  SvdStack m_svd_stack_left_dn{};
  SvdStack m_svd_stack_right_up{};
  SvdStack m_svd_stack_right_dn{};

  // pace of numerical stabilizations
  // or equivalently, the number of consequent wrapping steps of equal-time
  // greens functions
  int m_stabilization_pace{};

  // keep track of the wrapping error
  RealScalar m_wrap_error{};

  // ---------------------------------- Reweighting params
  // --------------------------------------- keep track of the sign problem
  RealScalar m_config_sign{};
  RealScalarVec m_vec_config_sign{};

 public:
  Walker() = default;

  // -------------------------------- Interfaces and friend class
  // --------------------------------

  int TimeSize() const { return this->m_time_size; }
  RealScalar Beta() const { return this->m_beta; }
  RealScalar TimeInterval() const { return this->m_time_interval; }
  RealScalar WrapError() const { return this->m_wrap_error; }
  int StabilizationPace() const { return this->m_stabilization_pace; }

  // interface for greens functions
  GreensFunc& GreenttUp() { return this->m_green_tt_up; }
  GreensFunc& GreenttDn() { return this->m_green_tt_dn; }
  const GreensFunc& GreenttUp() const { return this->m_green_tt_up; }
  const GreensFunc& GreenttDn() const { return this->m_green_tt_dn; }

  const GreensFunc& GreenttUp(int t) const { return this->m_vec_green_tt_up[t]; }
  const GreensFunc& GreenttDn(int t) const { return this->m_vec_green_tt_dn[t]; }
  const GreensFunc& Greent0Up(int t) const { return this->m_vec_green_t0_up[t]; }
  const GreensFunc& Greent0Dn(int t) const { return this->m_vec_green_t0_dn[t]; }
  const GreensFunc& Green0tUp(int t) const { return this->m_vec_green_0t_up[t]; }
  const GreensFunc& Green0tDn(int t) const { return this->m_vec_green_0t_dn[t]; }

  const GreensFuncVec& vecGreenttUp() const { return this->m_vec_green_tt_up; }
  const GreensFuncVec& vecGreenttDn() const { return this->m_vec_green_tt_dn; }
  const GreensFuncVec& vecGreent0Up() const { return this->m_vec_green_t0_up; }
  const GreensFuncVec& vecGreent0Dn() const { return this->m_vec_green_t0_dn; }
  const GreensFuncVec& vecGreen0tUp() const { return this->m_vec_green_0t_up; }
  const GreensFuncVec& vecGreen0tDn() const { return this->m_vec_green_0t_dn; }

  // interfaces for configuration signs
  RealScalar ConfigSign() const { return this->m_config_sign; }

  RealScalar ConfigSign(int t) const { return this->m_vec_config_sign[t]; }

  const RealScalarVec& vecConfigSign() const { return this->m_vec_config_sign; }

  // output MonteCarlo parameters information
  void output_montecarlo_info(std::ostream& ostream) const;

  // output imaginary-time grids
  void output_imaginary_time_grids(std::ostream& ostream) const;

  friend class Initializer;

  // ------------------------------- Setup of parameters
  // -----------------------------------------

  // set up the physical parameters
  // especially the inverse temperature and the number of time slices
  void set_physical_params(RealScalar beta, int time_size);

  // set up the pace of stabilizations
  void set_stabilization_pace(int stabilization_pace);

 private:
  // ---------------------------------- Initializations
  // ------------------------------------------ never explicitly call these
  // functions to avoid unpredictable mistakes, and use Initializer instead

  void initial(const LatticeBase& lattice, const MeasureHandler& meas_handler);

  void initial_svd_stacks(const LatticeBase& lattice, const ModelBase& model);

  // caution that this is a member function to initialize the model module
  // svd stacks should be initialized in advance
  void initial_greens_functions();

  // compute the sign of the initial bosonic configurations
  void initial_config_sign();

  // allocate memory
  void allocate_svd_stacks();
  void allocate_greens_functions();

 public:
  // ---------------------------------- Monte Carlo updates
  // --------------------------------------

  // sweep forwards from time slice 0 to beta
  void sweep_from_0_to_beta(ModelBase& model, std::default_random_engine& rng);

  // sweep backwards from time slice beta to 0
  void sweep_from_beta_to_0(ModelBase& model, std::default_random_engine& rng);

  // sweep backwards from beta to 0 especially to compute dynamic greens
  // functions, without the updates of bosonic fields
  void sweep_for_dynamic_greens(ModelBase& model);

 private:
  // update the bosonic fields at time slice t using Metropolis algorithm
  void metropolis_update(ModelBase& model, TimeIndex t, std::default_random_engine& rng);

  // wrap the equal-time greens functions from time slice t to t+1
  void wrap_from_0_to_beta(const ModelBase& model, TimeIndex t);

  // wrap the equal-time greens functions from time slice t to t-1
  void wrap_from_beta_to_0(const ModelBase& model, TimeIndex t);
};
}  // namespace DQMC
