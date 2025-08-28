#pragma once

/**
 *  This header file defines the crucial class DQMC::Walker
 *  to organize the entire dqmc program.
 */

#include <Eigen/Core>
#include <random>
#include <vector>

#include "svd_stack.h"
#include "utils/numerical_stable.hpp"

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
class Dqmc;

using LatticeBase = Lattice::LatticeBase;
using ModelBase = Model::ModelBase;
using MeasureHandler = Measure::MeasureHandler;

// ---------------------------- Crucial class DQMC::Walker
// ----------------------------
class Walker {
 private:
  using SvdStack = Utils::SVD_stack;
  using GreensFunc = Eigen::MatrixXd;

  // --------------------------------- Walker params
  // ---------------------------------------------

  int m_space_size{};          // number of space sites
  int m_time_size{};           // number of time slices
  double m_beta{};             // inverse temperature beta
  double m_time_interval{};    // interval of imaginary-time grids
  int m_current_time_slice{};  // helping params to record current time slice

  // ----------------------- ( Equal-time and dynamic ) Greens functions
  // -------------------------

  // Equal-time green's functions, which is the most crucial quantities during
  // dqmc simulations for spin-1/2 systems, we label the spin index with up and
  // down
  GreensFunc m_green_tt_up{}, m_green_tt_down{};
  std::vector<GreensFunc> m_vec_green_tt_up{}, m_vec_green_tt_down{};

  // Time-displaced green's functions G(t,0) and G(0,t)
  // important for time-displaced measurements of physical observables
  GreensFunc m_green_t0_up{}, m_green_t0_down{};
  GreensFunc m_green_0t_up{}, m_green_0t_down{};
  std::vector<GreensFunc> m_vec_green_t0_up{}, m_vec_green_t0_down{};
  std::vector<GreensFunc> m_vec_green_0t_up{}, m_vec_green_0t_down{};

  bool m_is_equaltime{};
  bool m_is_dynamic{};

  // ------------------------- SvdStack for numerical stabilization
  // ------------------------------

  // Utils::SvdStack class
  // for efficient svd decompositions and numerical stabilization
  SvdStack m_svd_stack_left_up{};
  SvdStack m_svd_stack_left_down{};
  SvdStack m_svd_stack_right_up{};
  SvdStack m_svd_stack_right_down{};

  Utils::GreensWorkspace m_greens_workspace{};

  // pace of numerical stabilizations
  // or equivalently, the number of consequent wrapping steps of equal-time
  // greens functions
  int m_stabilization_pace{};

  // keep track of the wrapping error
  double m_wrap_error{};

  // ---------------------------------- Reweighting params
  // --------------------------------------- keep track of the sign problem
  double m_config_sign{};
  Eigen::VectorXd m_vec_config_sign{};

 public:
  explicit Walker(double beta, int time_size, int stabilization_pace);

  Walker() = delete;
  Walker(const Walker&) = delete;
  Walker& operator=(const Walker&) = delete;
  Walker(Walker&&) = delete;
  Walker& operator=(Walker&&) = delete;

  // -------------------------------- Interfaces and friend class
  // --------------------------------

 public:
  int time_size() const { return this->m_time_size; }
  double beta() const { return this->m_beta; }
  double time_interval() const { return this->m_time_interval; }
  double wrap_error() const { return this->m_wrap_error; }
  int stabilization_pace() const { return this->m_stabilization_pace; }

  // interface for greens functions
  GreensFunc& green_tt_up() { return this->m_green_tt_up; }
  GreensFunc& green_tt_down() { return this->m_green_tt_down; }
  const GreensFunc& green_tt_up() const { return this->m_green_tt_up; }
  const GreensFunc& green_tt_down() const { return this->m_green_tt_down; }

  const GreensFunc& green_tt_up(int t) const { return this->m_vec_green_tt_up[t]; }
  const GreensFunc& green_tt_down(int t) const { return this->m_vec_green_tt_down[t]; }
  const GreensFunc& green_t0_up(int t) const { return this->m_vec_green_t0_up[t]; }
  const GreensFunc& green_t0_down(int t) const { return this->m_vec_green_t0_down[t]; }
  const GreensFunc& green_0t_up(int t) const { return this->m_vec_green_0t_up[t]; }
  const GreensFunc& green_0t_down(int t) const { return this->m_vec_green_0t_down[t]; }

  // interfaces for configuration signs
  double config_sign() const { return this->m_config_sign; }

  double config_sign(int t) const { return this->m_vec_config_sign[t]; }

  const Eigen::VectorXd& vec_config_sign() const { return this->m_vec_config_sign; }

  // output MonteCarlo parameters information
  void output_montecarlo_info(std::ostream& ostream) const;

  // output imaginary-time grids
  void output_imaginary_time_grids(std::ostream& ostream) const;

  friend class Initializer;
  friend class Dqmc;

 private:
  // ------------------------------- Setup of parameters (now private, called by constructor)
  // -----------------------------------------

  // set up the physical parameters
  // especially the inverse temperature and the number of time slices
  void set_physical_params(double beta, int time_size);

  // set up the pace of stabilizations
  void set_stabilization_pace(int stabilization_pace);
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
  void metropolis_update(ModelBase& model, int time_index, std::default_random_engine& rng);

  // wrap the equal-time greens functions from time slice t to t+1
  void wrap_from_0_to_beta(const ModelBase& model, int time_index);

  // wrap the equal-time greens functions from time slice t to t-1
  void wrap_from_beta_to_0(const ModelBase& model, int time_index);
};
}  // namespace DQMC
