#include "walker.h"

#include <format>
#include <random>

#include "lattice/lattice_base.h"
#include "measure/measure_handler.h"
#include "model/model_base.h"
#include "utils/assert.h"
#include "utils/numerical_stable.hpp"

namespace DQMC {

// alias conventions
using NumericalStable = Utils::NumericalStable;

Walker::Walker(double beta, int time_size, int stabilization_pace) {
  set_physical_params(beta, time_size);
  set_stabilization_pace(stabilization_pace);
}

void Walker::set_physical_params(double beta, int time_size) {
  DQMC_ASSERT(beta > 0.0);
  this->m_beta = beta;
  this->m_time_size = time_size;
  this->m_time_interval = beta / time_size;
}

void Walker::set_stabilization_pace(int stabilization_pace) {
  DQMC_ASSERT(stabilization_pace > 0);
  this->m_stabilization_pace = stabilization_pace;
}

void Walker::initial(const LatticeBase& lattice, const MeasureHandler& meas_handler) {
  this->m_space_size = lattice.space_size();
  this->m_current_time_slice = 0;
  this->m_wrap_error = 0.0;

  this->m_is_equaltime = meas_handler.is_equaltime();
  this->m_is_dynamic = meas_handler.is_dynamic();
}

void Walker::allocate_svd_stacks() {
  // allocate memory for SvdStack classes
  this->m_svd_stack_left_up = SvdStack(this->m_space_size, this->m_time_size);
  this->m_svd_stack_left_down = SvdStack(this->m_space_size, this->m_time_size);
  this->m_svd_stack_right_up = SvdStack(this->m_space_size, this->m_time_size);
  this->m_svd_stack_right_down = SvdStack(this->m_space_size, this->m_time_size);
}

void Walker::allocate_greens_functions() {
  // allocate memory for greens functions
  this->m_green_tt_up = GreensFunc(this->m_space_size, this->m_space_size);
  this->m_green_tt_down = GreensFunc(this->m_space_size, this->m_space_size);

  if (this->m_is_equaltime || this->m_is_dynamic) {
    this->m_vec_green_tt_up = std::vector<GreensFunc>(
        this->m_time_size, GreensFunc(this->m_space_size, this->m_space_size));
    this->m_vec_green_tt_down = std::vector<GreensFunc>(
        this->m_time_size, GreensFunc(this->m_space_size, this->m_space_size));
  }

  if (this->m_is_dynamic) {
    this->m_green_t0_up = GreensFunc(this->m_space_size, this->m_space_size);
    this->m_green_t0_down = GreensFunc(this->m_space_size, this->m_space_size);
    this->m_green_0t_up = GreensFunc(this->m_space_size, this->m_space_size);
    this->m_green_0t_down = GreensFunc(this->m_space_size, this->m_space_size);

    this->m_vec_green_t0_up = std::vector<GreensFunc>(
        this->m_time_size, GreensFunc(this->m_space_size, this->m_space_size));
    this->m_vec_green_t0_down = std::vector<GreensFunc>(
        this->m_time_size, GreensFunc(this->m_space_size, this->m_space_size));
    this->m_vec_green_0t_up = std::vector<GreensFunc>(
        this->m_time_size, GreensFunc(this->m_space_size, this->m_space_size));
    this->m_vec_green_0t_down = std::vector<GreensFunc>(
        this->m_time_size, GreensFunc(this->m_space_size, this->m_space_size));
  }
}

void Walker::initial_svd_stacks([[maybe_unused]] const LatticeBase& lattice,
                                const ModelBase& model) {
  // initialize udv stacks for sweep use
  // sweep process will start from 0 to beta, so we initialize svd_stack_right
  // here. stabilize the process every stabilization_pace steps

  // allocate memory
  this->allocate_svd_stacks();

  auto tmp_stack_up = m_tmp_pool.acquire_matrix(m_space_size, m_space_size);
  auto tmp_stack_down = m_tmp_pool.acquire_matrix(m_space_size, m_space_size);
  tmp_stack_up->setIdentity();
  tmp_stack_down->setIdentity();

  // initial svd stacks for sweeping usages
  for (auto time_index = this->m_time_size; time_index >= 1; --time_index) {
    model.mult_transB_from_left(*tmp_stack_up, time_index, +1.0);
    model.mult_transB_from_left(*tmp_stack_down, time_index, -1.0);

    // stabilize every nwrap steps with svd decomposition
    if ((time_index - 1) % this->m_stabilization_pace == 0) {
      this->m_svd_stack_right_up.push(*tmp_stack_up);
      this->m_svd_stack_right_down.push(*tmp_stack_down);
      tmp_stack_up->setIdentity();
      tmp_stack_down->setIdentity();
    }
  }
}

void Walker::initial_greens_functions() {
  // allocate memory
  this->allocate_greens_functions();

  // compute greens function at time slice t = 0
  // which corresponds to imaginary-time tau = beta
  // the svd stacks should be initialized correctly ahead of time
  NumericalStable::compute_equaltime_greens(this->m_svd_stack_left_up, this->m_svd_stack_right_up,
                                            this->m_green_tt_up, this->m_tmp_pool);
  NumericalStable::compute_equaltime_greens(this->m_svd_stack_left_down,
                                            this->m_svd_stack_right_down, this->m_green_tt_down,
                                            this->m_tmp_pool);
}

void Walker::initial_config_sign() {
  // allocate memory for config sign vector
  // if equal-time measurements are to be performed
  if (this->m_is_equaltime) {
    this->m_vec_config_sign = Eigen::VectorXd(this->m_time_size);
  }

  // initialize the sign of the initial bosonic configurations
  this->m_config_sign =
      (this->m_green_tt_up.determinant() * this->m_green_tt_down.determinant() >= 0) ? +1.0 : -1.0;
}

/*
 *  Update the aux bosonic fields at space-time position (t,i)
 *  for all i with Metropolis probability, and, if the update is accepted,
 *  perform a in-place update of the green's functions.
 *  Record the updated green's function at the life-end of this function.
 */
void Walker::metropolis_update(ModelBase& model, int time_index, std::default_random_engine& rng) {
  DQMC_ASSERT(this->m_current_time_slice == time_index);
  DQMC_ASSERT(time_index >= 0 && time_index <= this->m_time_size);

  const int effective_time_index = (time_index == 0) ? this->m_time_size - 1 : time_index - 1;
  for (auto i = 0; i < this->m_space_size; ++i) {
    // obtain the ratio of flipping the bosonic field at (i,l)
    const auto update_ratio = model.get_update_ratio(*this, effective_time_index, i);

    if (std::bernoulli_distribution(std::min(1.0, std::abs(update_ratio)))(rng)) {
      // if accepted update the greens functions
      model.update_greens_function(*this, effective_time_index, i);

      // update the bosonic fields
      model.update_bosonic_field(effective_time_index, i);

      // keep track of sign problem
      this->m_config_sign = (update_ratio >= 0) ? +this->m_config_sign : -this->m_config_sign;
    }
  }
}

/*
 *  Propagate the greens functions from the current time slice t
 *  forwards to the time slice t+1 according to
 *      G(t+1) = B(t+1) * G(t) * B(t+1)^-1
 *  for both spin-1/2 states.
 *  The greens functions are changed in place.
 */
void Walker::wrap_from_0_to_beta(const ModelBase& model, int time_index) {
  DQMC_ASSERT(time_index >= 0 && time_index <= this->m_time_size);

  const int effective_time_index = (time_index == this->m_time_size) ? 1 : time_index + 1;
  model.mult_B_from_left(this->m_green_tt_up, effective_time_index, +1);
  model.mult_invB_from_right(this->m_green_tt_up, effective_time_index, +1);
  model.mult_B_from_left(this->m_green_tt_down, effective_time_index, -1);
  model.mult_invB_from_right(this->m_green_tt_down, effective_time_index, -1);
}

/*
 *  Propagate the greens function from the current time slice t
 *  downwards to the time slice t-1 according to
 *      G(t-1) = B(t)^-1 * G(t) * B(t)
 *  for both spin-1/2 states.
 *  The greens functions are changed in place.
 */
void Walker::wrap_from_beta_to_0(const ModelBase& model, int time_index) {
  DQMC_ASSERT(time_index >= 0 && time_index <= this->m_time_size);

  const int effective_time_index = (time_index == 0) ? this->m_time_size : time_index;
  model.mult_B_from_right(this->m_green_tt_up, effective_time_index, +1);
  model.mult_invB_from_left(this->m_green_tt_up, effective_time_index, +1);
  model.mult_B_from_right(this->m_green_tt_down, effective_time_index, -1);
  model.mult_invB_from_left(this->m_green_tt_down, effective_time_index, -1);
}

/*
 *  Update the space-time lattice of the auxiliary bosonic fields.
 *  For t = 1,2...,ts , attempt to update fields and propagate the greens
 * functions Perform the stabilization every 'stabilization_pace' time slices
 */
void Walker::sweep_from_0_to_beta(ModelBase& model, std::default_random_engine& rng) {
  this->m_current_time_slice++;

  [[maybe_unused]] const int stack_length =
      (this->m_time_size % this->m_stabilization_pace == 0)
          ? this->m_time_size / this->m_stabilization_pace
          : this->m_time_size / this->m_stabilization_pace + 1;
  DQMC_ASSERT(this->m_current_time_slice == 1);
  DQMC_ASSERT(this->m_svd_stack_left_up.empty() && this->m_svd_stack_left_down.empty());
  DQMC_ASSERT(this->m_svd_stack_right_up.size() == stack_length &&
              this->m_svd_stack_right_down.size() == stack_length);

  // Use pre-allocated temporary matrices from workspace
  auto tmp_mat_up = m_tmp_pool.acquire_matrix(m_space_size, m_space_size);
  auto tmp_mat_down = m_tmp_pool.acquire_matrix(m_space_size, m_space_size);
  auto tmp_green_tt_up = m_tmp_pool.acquire_matrix(m_space_size, m_space_size);
  auto tmp_green_tt_down = m_tmp_pool.acquire_matrix(m_space_size, m_space_size);

  tmp_mat_up->setIdentity();
  tmp_mat_down->setIdentity();

  // sweep upwards from 0 to beta
  for (auto time_index = 1; time_index <= this->m_time_size; ++time_index) {
    // wrap green function to current time slice t
    this->wrap_from_0_to_beta(model, time_index - 1);

    // update auxiliary fields and record the updated greens functions
    this->metropolis_update(model, time_index, rng);
    if (this->m_is_equaltime) {
      this->m_vec_green_tt_up[time_index - 1] = this->m_green_tt_up;
      this->m_vec_green_tt_down[time_index - 1] = this->m_green_tt_down;
      this->m_vec_config_sign[time_index - 1] = this->m_config_sign;
    }

    model.mult_B_from_left(*tmp_mat_up, time_index, +1);
    model.mult_B_from_left(*tmp_mat_down, time_index, -1);

    // perform the stabilizations
    if (time_index % this->m_stabilization_pace == 0 || time_index == this->m_time_size) {
      // update svd stacks
      this->m_svd_stack_right_up.pop();
      this->m_svd_stack_right_down.pop();
      this->m_svd_stack_left_up.push(*tmp_mat_up);
      this->m_svd_stack_left_down.push(*tmp_mat_down);

      // collect the wrapping errors
      tmp_green_tt_up->setZero();
      tmp_green_tt_down->setZero();
      double tmp_wrap_error_tt_up = 0.0;
      double tmp_wrap_error_tt_down = 0.0;

      // compute fresh greens every 'stabilization_pace' steps: g = ( 1 +
      // stack_left*stack_right^T )^-1 stack_left = B(t-1) * ... * B(0)
      // stack_right = B(t)^T * ... * B(ts-1)^T
      NumericalStable::compute_equaltime_greens(this->m_svd_stack_left_up,
                                                this->m_svd_stack_right_up, *tmp_green_tt_up,
                                                this->m_tmp_pool);
      NumericalStable::compute_equaltime_greens(this->m_svd_stack_left_down,
                                                this->m_svd_stack_right_down, *tmp_green_tt_down,
                                                this->m_tmp_pool);

      // compute wrapping errors
      NumericalStable::matrix_compare_error(*tmp_green_tt_up, this->m_green_tt_up,
                                            tmp_wrap_error_tt_up);
      NumericalStable::matrix_compare_error(*tmp_green_tt_down, this->m_green_tt_down,
                                            tmp_wrap_error_tt_down);
      this->m_wrap_error =
          std::max(this->m_wrap_error, std::max(tmp_wrap_error_tt_up, tmp_wrap_error_tt_down));

      this->m_green_tt_up = *tmp_green_tt_up;
      this->m_green_tt_down = *tmp_green_tt_down;

      if (this->m_is_equaltime) {
        this->m_vec_green_tt_up[time_index - 1] = this->m_green_tt_up;
        this->m_vec_green_tt_down[time_index - 1] = this->m_green_tt_down;
      }

      tmp_mat_up->setIdentity();
      tmp_mat_down->setIdentity();
    }

    // finally stop at time slice t = ts + 1
    this->m_current_time_slice++;
  }

  // end with fresh greens functions
  if (this->m_is_equaltime) {
    this->m_vec_green_tt_up[this->m_time_size - 1] = this->m_green_tt_up;
    this->m_vec_green_tt_down[this->m_time_size - 1] = this->m_green_tt_down;
  }
}

/*
 *  Update the space-time lattice of the auxiliary bosonic fields.
 *  For l = ts,ts-1,...,1 , attempt to update fields and propagate the greens
 * functions Perform the stabilization every 'stabilization_pace' time slices
 */
void Walker::sweep_from_beta_to_0(ModelBase& model, std::default_random_engine& rng) {
  this->m_current_time_slice--;

  [[maybe_unused]] const int stack_length =
      (this->m_time_size % this->m_stabilization_pace == 0)
          ? this->m_time_size / this->m_stabilization_pace
          : this->m_time_size / this->m_stabilization_pace + 1;
  DQMC_ASSERT(this->m_current_time_slice == this->m_time_size);
  DQMC_ASSERT(this->m_svd_stack_right_up.empty() && this->m_svd_stack_right_down.empty());
  DQMC_ASSERT(this->m_svd_stack_left_up.size() == stack_length &&
              this->m_svd_stack_left_down.size() == stack_length);

  // Use pre-allocated temporary matrices from workspace
  auto tmp_mat_up = m_tmp_pool.acquire_matrix(m_space_size, m_space_size);
  auto tmp_mat_down = m_tmp_pool.acquire_matrix(m_space_size, m_space_size);
  auto tmp_green_tt_up = m_tmp_pool.acquire_matrix(m_space_size, m_space_size);
  auto tmp_green_tt_down = m_tmp_pool.acquire_matrix(m_space_size, m_space_size);

  tmp_mat_up->setIdentity();
  tmp_mat_down->setIdentity();

  // sweep downwards from beta to 0
  for (auto time_index = this->m_time_size; time_index >= 1; --time_index) {
    // perform the stabilizations
    if (time_index % this->m_stabilization_pace == 0 && time_index != this->m_time_size) {
      // update svd stacks
      this->m_svd_stack_left_up.pop();
      this->m_svd_stack_left_down.pop();
      this->m_svd_stack_right_up.push(*tmp_mat_up);
      this->m_svd_stack_right_down.push(*tmp_mat_down);

      // collect the wrapping errors
      tmp_green_tt_up->setZero();
      tmp_green_tt_down->setZero();
      double tmp_wrap_error_tt_up = 0.0;
      double tmp_wrap_error_tt_down = 0.0;

      NumericalStable::compute_equaltime_greens(this->m_svd_stack_left_up,
                                                this->m_svd_stack_right_up, *tmp_green_tt_up,
                                                this->m_tmp_pool);
      NumericalStable::compute_equaltime_greens(this->m_svd_stack_left_down,
                                                this->m_svd_stack_right_down, *tmp_green_tt_down,
                                                this->m_tmp_pool);

      // compute the wrapping errors
      NumericalStable::matrix_compare_error(*tmp_green_tt_up, this->m_green_tt_up,
                                            tmp_wrap_error_tt_up);
      NumericalStable::matrix_compare_error(*tmp_green_tt_down, this->m_green_tt_down,
                                            tmp_wrap_error_tt_down);
      this->m_wrap_error =
          std::max(this->m_wrap_error, std::max(tmp_wrap_error_tt_up, tmp_wrap_error_tt_down));

      this->m_green_tt_up = *tmp_green_tt_up;
      this->m_green_tt_down = *tmp_green_tt_down;

      tmp_mat_up->setIdentity();
      tmp_mat_down->setIdentity();
    }

    // update auxiliary fields and record the updated greens functions
    this->metropolis_update(model, time_index, rng);
    if (this->m_is_equaltime) {
      this->m_vec_green_tt_up[time_index - 1] = this->m_green_tt_up;
      this->m_vec_green_tt_down[time_index - 1] = this->m_green_tt_down;
      this->m_vec_config_sign[time_index - 1] = this->m_config_sign;
    }

    model.mult_transB_from_left(*tmp_mat_up, time_index, +1);
    model.mult_transB_from_left(*tmp_mat_down, time_index, -1);

    this->wrap_from_beta_to_0(model, time_index);

    this->m_current_time_slice--;
  }

  // at time slice t = 0
  this->m_svd_stack_left_up.pop();
  this->m_svd_stack_left_down.pop();
  this->m_svd_stack_right_up.push(*tmp_mat_up);
  this->m_svd_stack_right_down.push(*tmp_mat_down);

  NumericalStable::compute_equaltime_greens(this->m_svd_stack_left_up, this->m_svd_stack_right_up,
                                            this->m_green_tt_up, this->m_tmp_pool);
  NumericalStable::compute_equaltime_greens(this->m_svd_stack_left_down,
                                            this->m_svd_stack_right_down, this->m_green_tt_down,
                                            this->m_tmp_pool);

  // end with fresh greens functions
  if (this->m_is_equaltime) {
    this->m_vec_green_tt_up[this->m_time_size - 1] = this->m_green_tt_up;
    this->m_vec_green_tt_down[this->m_time_size - 1] = this->m_green_tt_down;
  }
}

/*
 *  Calculate time-displaced (dynamical) greens functions, while the auxiliary
 * fields remain unchanged. For l = 1,2...,ts , recompute the SvdStacks every
 * 'stabilization_pace' time slices. The collected dynamic greens functions are
 * stored in m_vec_green_t0(0t)_up(dn). Note that the equal-time greens
 * functions are also re-calculated according to the current auxiliary field
 * configurations, which are stored in m_vec_green_tt_up(dn).
 */
void Walker::sweep_for_dynamic_greens(ModelBase& model) {
  if (this->m_is_dynamic) {
    this->m_current_time_slice++;
    [[maybe_unused]] const int stack_length =
        (this->m_time_size % this->m_stabilization_pace == 0)
            ? this->m_time_size / this->m_stabilization_pace
            : this->m_time_size / this->m_stabilization_pace + 1;
    DQMC_ASSERT(this->m_current_time_slice == 1);
    DQMC_ASSERT(this->m_svd_stack_left_up.empty() && this->m_svd_stack_left_down.empty());
    DQMC_ASSERT(this->m_svd_stack_right_up.size() == stack_length &&
                this->m_svd_stack_right_down.size() == stack_length);

    // initialize greens functions: at t = 0, gt0 = g00, g0t = g00 - 1
    this->m_green_t0_up = this->m_green_tt_up;
    this->m_green_t0_down = this->m_green_tt_down;
    this->m_green_0t_up =
        this->m_green_tt_up - Eigen::MatrixXd::Identity(this->m_space_size, this->m_space_size);
    this->m_green_0t_down =
        this->m_green_tt_down - Eigen::MatrixXd::Identity(this->m_space_size, this->m_space_size);

    // Use pre-allocated temporary matrices from workspace
    auto tmp_mat_up = m_tmp_pool.acquire_matrix(m_space_size, m_space_size);
    auto tmp_mat_down = m_tmp_pool.acquire_matrix(m_space_size, m_space_size);
    auto tmp_green_t0_up = m_tmp_pool.acquire_matrix(m_space_size, m_space_size);
    auto tmp_green_t0_down = m_tmp_pool.acquire_matrix(m_space_size, m_space_size);
    auto tmp_green_0t_up = m_tmp_pool.acquire_matrix(m_space_size, m_space_size);
    auto tmp_green_0t_down = m_tmp_pool.acquire_matrix(m_space_size, m_space_size);

    tmp_mat_up->setIdentity();
    tmp_mat_down->setIdentity();

    // sweep forwards from 0 to beta
    for (auto time_index = 1; time_index <= this->m_time_size; ++time_index) {
      // wrap the equal time greens functions to current time slice t
      this->wrap_from_0_to_beta(model, time_index - 1);
      this->m_vec_green_tt_up[time_index - 1] = this->m_green_tt_up;
      this->m_vec_green_tt_down[time_index - 1] = this->m_green_tt_down;

      // calculate and record the time-displaced greens functions at different
      // time slices
      model.mult_B_from_left(this->m_green_t0_up, time_index, +1);
      model.mult_B_from_left(this->m_green_t0_down, time_index, -1);
      this->m_vec_green_t0_up[time_index - 1] = this->m_green_t0_up;
      this->m_vec_green_t0_down[time_index - 1] = this->m_green_t0_down;

      model.mult_invB_from_right(this->m_green_0t_up, time_index, +1);
      model.mult_invB_from_right(this->m_green_0t_down, time_index, -1);
      this->m_vec_green_0t_up[time_index - 1] = this->m_green_0t_up;
      this->m_vec_green_0t_down[time_index - 1] = this->m_green_0t_down;

      model.mult_B_from_left(*tmp_mat_up, time_index, +1);
      model.mult_B_from_left(*tmp_mat_down, time_index, -1);

      // perform the stabilizations
      if (time_index % this->m_stabilization_pace == 0 || time_index == this->m_time_size) {
        // update svd stacks
        this->m_svd_stack_right_up.pop();
        this->m_svd_stack_right_down.pop();
        this->m_svd_stack_left_up.push(*tmp_mat_up);
        this->m_svd_stack_left_down.push(*tmp_mat_down);

        // collect the wrapping errors
        tmp_green_t0_up->setZero();
        tmp_green_t0_down->setZero();
        tmp_green_0t_up->setZero();
        tmp_green_0t_down->setZero();
        double tmp_wrap_error_t0_up = 0.0;
        double tmp_wrap_error_t0_down = 0.0;
        double tmp_wrap_error_0t_up = 0.0;
        double tmp_wrap_error_0t_down = 0.0;

        // compute fresh greens every nwrap steps
        // stack_left = B(t-1) * ... * B(0)
        // stack_right = B(t)^T * ... * B(ts-1)^T
        // equal time green's function are re-evaluated for current field
        // configurations
        NumericalStable::compute_equaltime_greens(this->m_svd_stack_left_up,
                                                  this->m_svd_stack_right_up, this->m_green_tt_up,
                                                  this->m_tmp_pool);
        NumericalStable::compute_equaltime_greens(this->m_svd_stack_left_down,
                                                  this->m_svd_stack_right_down,
                                                  this->m_green_tt_down, this->m_tmp_pool);
        NumericalStable::compute_dynamic_greens(this->m_svd_stack_left_up,
                                                this->m_svd_stack_right_up, *tmp_green_t0_up,
                                                *tmp_green_0t_up, this->m_tmp_pool);
        NumericalStable::compute_dynamic_greens(this->m_svd_stack_left_down,
                                                this->m_svd_stack_right_down, *tmp_green_t0_down,
                                                *tmp_green_0t_down, this->m_tmp_pool);

        // compute wrapping errors
        NumericalStable::matrix_compare_error(*tmp_green_t0_up, this->m_green_t0_up,
                                              tmp_wrap_error_t0_up);
        NumericalStable::matrix_compare_error(*tmp_green_t0_down, this->m_green_t0_down,
                                              tmp_wrap_error_t0_down);
        this->m_wrap_error =
            std::max(this->m_wrap_error, std::max(tmp_wrap_error_t0_up, tmp_wrap_error_t0_down));

        NumericalStable::matrix_compare_error(*tmp_green_0t_up, this->m_green_0t_up,
                                              tmp_wrap_error_0t_up);
        NumericalStable::matrix_compare_error(*tmp_green_0t_down, this->m_green_0t_down,
                                              tmp_wrap_error_0t_down);
        this->m_wrap_error =
            std::max(this->m_wrap_error, std::max(tmp_wrap_error_0t_up, tmp_wrap_error_0t_down));

        this->m_green_t0_up = *tmp_green_t0_up;
        this->m_green_t0_down = *tmp_green_t0_down;
        this->m_green_0t_up = *tmp_green_0t_up;
        this->m_green_0t_down = *tmp_green_0t_down;

        this->m_vec_green_tt_up[time_index - 1] = this->m_green_tt_up;
        this->m_vec_green_tt_down[time_index - 1] = this->m_green_tt_down;
        this->m_vec_green_t0_up[time_index - 1] = this->m_green_t0_up;
        this->m_vec_green_t0_down[time_index - 1] = this->m_green_t0_down;
        this->m_vec_green_0t_up[time_index - 1] = this->m_green_0t_up;
        this->m_vec_green_0t_down[time_index - 1] = this->m_green_0t_down;

        tmp_mat_up->setIdentity();
        tmp_mat_down->setIdentity();
      }

      // finally stop at time slice t = ts + 1
      this->m_current_time_slice++;
    }
  }
}

void Walker::output_montecarlo_info(std::ostream& ostream) const {
  auto fmt_double = [](const std::string& desc, double value) {
    return std::format("{:>30s}{:>7s}{:>24.3f}\n", desc, "->", value);
  };

  auto fmt_int = [](const std::string& desc, int value) {
    return std::format("{:>30s}{:>7s}{:>24d}\n", desc, "->", value);
  };

  ostream << "   MonteCarlo Params:\n"
          << fmt_double("Inverse temperature", this->beta())
          << fmt_int("Imaginary-time length", this->time_size())
          << fmt_double("Imaginary-time interval", this->time_interval())
          << fmt_int("Stabilization pace", this->stabilization_pace()) << std::endl;
}

void Walker::output_imaginary_time_grids(std::ostream& ostream) const {
  // output the imaginary-time grids
  ostream << std::format("{:>20d}{:>20.5f}{:>20.5f}\n", this->time_size(), this->beta(),
                         this->time_interval());
  for (auto t = 0; t < this->time_size(); ++t) {
    ostream << std::format("{:>20d}{:>20.10f}\n", t, (t * this->time_interval()));
  }
}

}  // namespace DQMC
