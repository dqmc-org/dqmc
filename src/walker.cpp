#include "walker.h"

#include <format>
#include <random>

#include "lattice/lattice_base.h"
#include "measure/measure_handler.h"
#include "model/model_base.h"
#include "svd_stack.h"
#include "utils/assert.h"
#include "utils/numerical_stable.hpp"

namespace DQMC {

// alias conventions
using Matrix = Eigen::MatrixXd;
using NumericalStable = Utils::NumericalStable;

Walker::Walker(RealScalar beta, int time_size, int stabilization_pace) {
  set_physical_params(beta, time_size);
  set_stabilization_pace(stabilization_pace);
}

void Walker::set_physical_params(RealScalar beta, int time_size) {
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
  this->m_space_size = lattice.SpaceSize();
  this->m_current_time_slice = 0;
  this->m_wrap_error = 0.0;

  this->m_is_equaltime = meas_handler.isEqualTime();
  this->m_is_dynamic = meas_handler.isDynamic();
}

void Walker::allocate_svd_stacks() {
  // allocate memory for SvdStack classes
  this->m_svd_stack_left_up = SvdStack(this->m_space_size, this->m_time_size);
  this->m_svd_stack_left_dn = SvdStack(this->m_space_size, this->m_time_size);
  this->m_svd_stack_right_up = SvdStack(this->m_space_size, this->m_time_size);
  this->m_svd_stack_right_dn = SvdStack(this->m_space_size, this->m_time_size);
}

void Walker::allocate_greens_functions() {
  // allocate memory for greens functions
  this->m_green_tt_up = GreensFunc(this->m_space_size, this->m_space_size);
  this->m_green_tt_dn = GreensFunc(this->m_space_size, this->m_space_size);

  if (this->m_is_equaltime || this->m_is_dynamic) {
    this->m_vec_green_tt_up =
        GreensFuncVec(this->m_time_size, GreensFunc(this->m_space_size, this->m_space_size));
    this->m_vec_green_tt_dn =
        GreensFuncVec(this->m_time_size, GreensFunc(this->m_space_size, this->m_space_size));
  }

  if (this->m_is_dynamic) {
    this->m_green_t0_up = GreensFunc(this->m_space_size, this->m_space_size);
    this->m_green_t0_dn = GreensFunc(this->m_space_size, this->m_space_size);
    this->m_green_0t_up = GreensFunc(this->m_space_size, this->m_space_size);
    this->m_green_0t_dn = GreensFunc(this->m_space_size, this->m_space_size);

    this->m_vec_green_t0_up =
        GreensFuncVec(this->m_time_size, GreensFunc(this->m_space_size, this->m_space_size));
    this->m_vec_green_t0_dn =
        GreensFuncVec(this->m_time_size, GreensFunc(this->m_space_size, this->m_space_size));
    this->m_vec_green_0t_up =
        GreensFuncVec(this->m_time_size, GreensFunc(this->m_space_size, this->m_space_size));
    this->m_vec_green_0t_dn =
        GreensFuncVec(this->m_time_size, GreensFunc(this->m_space_size, this->m_space_size));
  }
}

void Walker::initial_svd_stacks([[maybe_unused]] const LatticeBase& lattice,
                                const ModelBase& model) {
  // initialize udv stacks for sweep use
  // sweep process will start from 0 to beta, so we initialize svd_stack_right
  // here. stabilize the process every stabilization_pace steps

  // allocate memory
  this->allocate_svd_stacks();

  Matrix tmp_stack_up = Matrix::Identity(this->m_space_size, this->m_space_size);
  Matrix tmp_stack_dn = Matrix::Identity(this->m_space_size, this->m_space_size);

  // initial svd stacks for sweeping usages
  for (auto t = this->m_time_size; t >= 1; --t) {
    model.mult_transB_from_left(tmp_stack_up, t, +1.0);
    model.mult_transB_from_left(tmp_stack_dn, t, -1.0);

    // stabilize every nwrap steps with svd decomposition
    if ((t - 1) % this->m_stabilization_pace == 0) {
      this->m_svd_stack_right_up.push(tmp_stack_up);
      this->m_svd_stack_right_dn.push(tmp_stack_dn);
      tmp_stack_up.setIdentity();
      tmp_stack_dn.setIdentity();
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
                                            this->m_green_tt_up, this->m_greens_workspace);
  NumericalStable::compute_equaltime_greens(this->m_svd_stack_left_dn, this->m_svd_stack_right_dn,
                                            this->m_green_tt_dn, this->m_greens_workspace);
}

void Walker::initial_config_sign() {
  // allocate memory for config sign vector
  // if equal-time measurements are to be performed
  if (this->m_is_equaltime) {
    this->m_vec_config_sign = RealScalarVec(this->m_time_size);
  }

  // initialize the sign of the initial bosonic configurations
  this->m_config_sign =
      (this->m_green_tt_up.determinant() * this->m_green_tt_dn.determinant() >= 0) ? +1.0 : -1.0;
}

/*
 *  Update the aux bosonic fields at space-time position (t,i)
 *  for all i with Metropolis probability, and, if the update is accepted,
 *  perform a in-place update of the green's functions.
 *  Record the updated green's function at the life-end of this function.
 */
void Walker::metropolis_update(ModelBase& model, TimeIndex t, std::default_random_engine& rng) {
  DQMC_ASSERT(this->m_current_time_slice == t);
  DQMC_ASSERT(t >= 0 && t <= this->m_time_size);

  const int eff_t = (t == 0) ? this->m_time_size - 1 : t - 1;
  for (auto i = 0; i < this->m_space_size; ++i) {
    // obtain the ratio of flipping the bosonic field at (i,l)
    const auto update_ratio = model.get_update_ratio(*this, eff_t, i);

    if (std::bernoulli_distribution(std::min(1.0, std::abs(update_ratio)))(rng)) {
      // if accepted
      // update the greens functions
      model.update_greens_function(*this, eff_t, i);

      // update the bosonic fields
      model.update_bosonic_field(eff_t, i);

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
void Walker::wrap_from_0_to_beta(const ModelBase& model, TimeIndex t) {
  DQMC_ASSERT(t >= 0 && t <= this->m_time_size);

  const int eff_t = (t == this->m_time_size) ? 1 : t + 1;
  model.mult_B_from_left(this->m_green_tt_up, eff_t, +1);
  model.mult_invB_from_right(this->m_green_tt_up, eff_t, +1);
  model.mult_B_from_left(this->m_green_tt_dn, eff_t, -1);
  model.mult_invB_from_right(this->m_green_tt_dn, eff_t, -1);
}

/*
 *  Propagate the greens function from the current time slice t
 *  downwards to the time slice t-1 according to
 *      G(t-1) = B(t)^-1 * G(t) * B(t)
 *  for both spin-1/2 states.
 *  The greens functions are changed in place.
 */
void Walker::wrap_from_beta_to_0(const ModelBase& model, TimeIndex t) {
  DQMC_ASSERT(t >= 0 && t <= this->m_time_size);

  const int eff_t = (t == 0) ? this->m_time_size : t;
  model.mult_B_from_right(this->m_green_tt_up, eff_t, +1);
  model.mult_invB_from_left(this->m_green_tt_up, eff_t, +1);
  model.mult_B_from_right(this->m_green_tt_dn, eff_t, -1);
  model.mult_invB_from_left(this->m_green_tt_dn, eff_t, -1);
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
  DQMC_ASSERT(this->m_svd_stack_left_up.empty() && this->m_svd_stack_left_dn.empty());
  DQMC_ASSERT(this->m_svd_stack_right_up.StackLength() == stack_length &&
              this->m_svd_stack_right_dn.StackLength() == stack_length);

  // temporary matrices
  Matrix tmp_mat_up = Matrix::Identity(this->m_space_size, this->m_space_size);
  Matrix tmp_mat_dn = Matrix::Identity(this->m_space_size, this->m_space_size);
  Matrix tmp_green_tt_up(this->m_space_size, this->m_space_size);
  Matrix tmp_green_tt_dn(this->m_space_size, this->m_space_size);

  // sweep upwards from 0 to beta
  for (auto t = 1; t <= this->m_time_size; ++t) {
    // wrap green function to current time slice t
    this->wrap_from_0_to_beta(model, t - 1);

    // update auxiliary fields and record the updated greens functions
    this->metropolis_update(model, t, rng);
    if (this->m_is_equaltime) {
      this->m_vec_green_tt_up[t - 1] = this->m_green_tt_up;
      this->m_vec_green_tt_dn[t - 1] = this->m_green_tt_dn;
      this->m_vec_config_sign[t - 1] = this->m_config_sign;
    }

    model.mult_B_from_left(tmp_mat_up, t, +1);
    model.mult_B_from_left(tmp_mat_dn, t, -1);

    // perform the stabilizations
    if (t % this->m_stabilization_pace == 0 || t == this->m_time_size) {
      // update svd stacks
      this->m_svd_stack_right_up.pop();
      this->m_svd_stack_right_dn.pop();
      this->m_svd_stack_left_up.push(tmp_mat_up);
      this->m_svd_stack_left_dn.push(tmp_mat_dn);

      // collect the wrapping errors
      tmp_green_tt_up.setZero();
      tmp_green_tt_dn.setZero();
      double tmp_wrap_error_tt_up = 0.0;
      double tmp_wrap_error_tt_dn = 0.0;

      // compute fresh greens every 'stabilization_pace' steps: g = ( 1 +
      // stack_left*stack_right^T )^-1 stack_left = B(t-1) * ... * B(0)
      // stack_right = B(t)^T * ... * B(ts-1)^T
      NumericalStable::compute_equaltime_greens(this->m_svd_stack_left_up,
                                                this->m_svd_stack_right_up, tmp_green_tt_up,
                                                this->m_greens_workspace);
      NumericalStable::compute_equaltime_greens(this->m_svd_stack_left_dn,
                                                this->m_svd_stack_right_dn, tmp_green_tt_dn,
                                                this->m_greens_workspace);

      // compute wrapping errors
      NumericalStable::matrix_compare_error(tmp_green_tt_up, this->m_green_tt_up,
                                            tmp_wrap_error_tt_up);
      NumericalStable::matrix_compare_error(tmp_green_tt_dn, this->m_green_tt_dn,
                                            tmp_wrap_error_tt_dn);
      this->m_wrap_error =
          std::max(this->m_wrap_error, std::max(tmp_wrap_error_tt_up, tmp_wrap_error_tt_dn));

      this->m_green_tt_up = tmp_green_tt_up;
      this->m_green_tt_dn = tmp_green_tt_dn;

      if (this->m_is_equaltime) {
        this->m_vec_green_tt_up[t - 1] = this->m_green_tt_up;
        this->m_vec_green_tt_dn[t - 1] = this->m_green_tt_dn;
      }

      tmp_mat_up.setIdentity();
      tmp_mat_dn.setIdentity();
    }

    // finally stop at time slice t = ts + 1
    this->m_current_time_slice++;
  }

  // end with fresh greens functions
  if (this->m_is_equaltime) {
    this->m_vec_green_tt_up[this->m_time_size - 1] = this->m_green_tt_up;
    this->m_vec_green_tt_dn[this->m_time_size - 1] = this->m_green_tt_dn;
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
  DQMC_ASSERT(this->m_svd_stack_right_up.empty() && this->m_svd_stack_right_dn.empty());
  DQMC_ASSERT(this->m_svd_stack_left_up.StackLength() == stack_length &&
              this->m_svd_stack_left_dn.StackLength() == stack_length);

  // temporary matrices
  Matrix tmp_mat_up = Matrix::Identity(this->m_space_size, this->m_space_size);
  Matrix tmp_mat_dn = Matrix::Identity(this->m_space_size, this->m_space_size);
  Matrix tmp_green_tt_up(this->m_space_size, this->m_space_size);
  Matrix tmp_green_tt_dn(this->m_space_size, this->m_space_size);

  // sweep downwards from beta to 0
  for (auto t = this->m_time_size; t >= 1; --t) {
    // perform the stabilizations
    if (t % this->m_stabilization_pace == 0 && t != this->m_time_size) {
      // update svd stacks
      this->m_svd_stack_left_up.pop();
      this->m_svd_stack_left_dn.pop();
      this->m_svd_stack_right_up.push(tmp_mat_up);
      this->m_svd_stack_right_dn.push(tmp_mat_dn);

      // collect the wrapping errors
      tmp_green_tt_up.setZero();
      tmp_green_tt_dn.setZero();
      double tmp_wrap_error_tt_up = 0.0;
      double tmp_wrap_error_tt_dn = 0.0;

      NumericalStable::compute_equaltime_greens(this->m_svd_stack_left_up,
                                                this->m_svd_stack_right_up, tmp_green_tt_up,
                                                this->m_greens_workspace);
      NumericalStable::compute_equaltime_greens(this->m_svd_stack_left_dn,
                                                this->m_svd_stack_right_dn, tmp_green_tt_dn,
                                                this->m_greens_workspace);

      // compute the wrapping errors
      NumericalStable::matrix_compare_error(tmp_green_tt_up, this->m_green_tt_up,
                                            tmp_wrap_error_tt_up);
      NumericalStable::matrix_compare_error(tmp_green_tt_dn, this->m_green_tt_dn,
                                            tmp_wrap_error_tt_dn);
      this->m_wrap_error =
          std::max(this->m_wrap_error, std::max(tmp_wrap_error_tt_up, tmp_wrap_error_tt_dn));

      this->m_green_tt_up = tmp_green_tt_up;
      this->m_green_tt_dn = tmp_green_tt_dn;

      tmp_mat_up.setIdentity();
      tmp_mat_dn.setIdentity();
    }

    // update auxiliary fields and record the updated greens functions
    this->metropolis_update(model, t, rng);
    if (this->m_is_equaltime) {
      this->m_vec_green_tt_up[t - 1] = this->m_green_tt_up;
      this->m_vec_green_tt_dn[t - 1] = this->m_green_tt_dn;
      this->m_vec_config_sign[t - 1] = this->m_config_sign;
    }

    model.mult_transB_from_left(tmp_mat_up, t, +1);
    model.mult_transB_from_left(tmp_mat_dn, t, -1);

    this->wrap_from_beta_to_0(model, t);

    this->m_current_time_slice--;
  }

  // at time slice t = 0
  this->m_svd_stack_left_up.pop();
  this->m_svd_stack_left_dn.pop();
  this->m_svd_stack_right_up.push(tmp_mat_up);
  this->m_svd_stack_right_dn.push(tmp_mat_dn);

  NumericalStable::compute_equaltime_greens(this->m_svd_stack_left_up, this->m_svd_stack_right_up,
                                            this->m_green_tt_up, this->m_greens_workspace);
  NumericalStable::compute_equaltime_greens(this->m_svd_stack_left_dn, this->m_svd_stack_right_dn,
                                            this->m_green_tt_dn, this->m_greens_workspace);

  // end with fresh greens functions
  if (this->m_is_equaltime) {
    this->m_vec_green_tt_up[this->m_time_size - 1] = this->m_green_tt_up;
    this->m_vec_green_tt_dn[this->m_time_size - 1] = this->m_green_tt_dn;
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
    DQMC_ASSERT(this->m_svd_stack_left_up.empty() && this->m_svd_stack_left_dn.empty());
    DQMC_ASSERT(this->m_svd_stack_right_up.StackLength() == stack_length &&
                this->m_svd_stack_right_dn.StackLength() == stack_length);

    // initialize greens functions: at t = 0, gt0 = g00, g0t = g00 - 1
    this->m_green_t0_up = this->m_green_tt_up;
    this->m_green_t0_dn = this->m_green_tt_dn;
    this->m_green_0t_up =
        this->m_green_tt_up - Matrix::Identity(this->m_space_size, this->m_space_size);
    this->m_green_0t_dn =
        this->m_green_tt_dn - Matrix::Identity(this->m_space_size, this->m_space_size);

    // temporary matrices
    Matrix tmp_mat_up = Matrix::Identity(this->m_space_size, this->m_space_size);
    Matrix tmp_mat_dn = Matrix::Identity(this->m_space_size, this->m_space_size);
    Matrix tmp_green_t0_up(this->m_space_size, this->m_space_size);
    Matrix tmp_green_t0_dn(this->m_space_size, this->m_space_size);
    Matrix tmp_green_0t_up(this->m_space_size, this->m_space_size);
    Matrix tmp_green_0t_dn(this->m_space_size, this->m_space_size);

    // sweep forwards from 0 to beta
    for (auto t = 1; t <= this->m_time_size; ++t) {
      // wrap the equal time greens functions to current time slice t
      this->wrap_from_0_to_beta(model, t - 1);
      this->m_vec_green_tt_up[t - 1] = this->m_green_tt_up;
      this->m_vec_green_tt_dn[t - 1] = this->m_green_tt_dn;

      // calculate and record the time-displaced greens functions at different
      // time slices
      model.mult_B_from_left(this->m_green_t0_up, t, +1);
      model.mult_B_from_left(this->m_green_t0_dn, t, -1);
      this->m_vec_green_t0_up[t - 1] = this->m_green_t0_up;
      this->m_vec_green_t0_dn[t - 1] = this->m_green_t0_dn;

      model.mult_invB_from_right(this->m_green_0t_up, t, +1);
      model.mult_invB_from_right(this->m_green_0t_dn, t, -1);
      this->m_vec_green_0t_up[t - 1] = this->m_green_0t_up;
      this->m_vec_green_0t_dn[t - 1] = this->m_green_0t_dn;

      model.mult_B_from_left(tmp_mat_up, t, +1);
      model.mult_B_from_left(tmp_mat_dn, t, -1);

      // perform the stabilizations
      if (t % this->m_stabilization_pace == 0 || t == this->m_time_size) {
        // update svd stacks
        this->m_svd_stack_right_up.pop();
        this->m_svd_stack_right_dn.pop();
        this->m_svd_stack_left_up.push(tmp_mat_up);
        this->m_svd_stack_left_dn.push(tmp_mat_dn);

        // collect the wrapping errors
        tmp_green_t0_up.setZero();
        tmp_green_t0_dn.setZero();
        tmp_green_0t_up.setZero();
        tmp_green_0t_dn.setZero();
        double tmp_wrap_error_t0_up = 0.0;
        double tmp_wrap_error_t0_dn = 0.0;
        double tmp_wrap_error_0t_up = 0.0;
        double tmp_wrap_error_0t_dn = 0.0;

        // compute fresh greens every nwrap steps
        // stack_left = B(t-1) * ... * B(0)
        // stack_right = B(t)^T * ... * B(ts-1)^T
        // equal time green's function are re-evaluated for current field
        // configurations
        NumericalStable::compute_equaltime_greens(this->m_svd_stack_left_up,
                                                  this->m_svd_stack_right_up, this->m_green_tt_up,
                                                  this->m_greens_workspace);
        NumericalStable::compute_equaltime_greens(this->m_svd_stack_left_dn,
                                                  this->m_svd_stack_right_dn, this->m_green_tt_dn,
                                                  this->m_greens_workspace);
        NumericalStable::compute_dynamic_greens(this->m_svd_stack_left_up,
                                                this->m_svd_stack_right_up, tmp_green_t0_up,
                                                tmp_green_0t_up, this->m_greens_workspace);
        NumericalStable::compute_dynamic_greens(this->m_svd_stack_left_dn,
                                                this->m_svd_stack_right_dn, tmp_green_t0_dn,
                                                tmp_green_0t_dn, this->m_greens_workspace);

        // compute wrapping errors
        NumericalStable::matrix_compare_error(tmp_green_t0_up, this->m_green_t0_up,
                                              tmp_wrap_error_t0_up);
        NumericalStable::matrix_compare_error(tmp_green_t0_dn, this->m_green_t0_dn,
                                              tmp_wrap_error_t0_dn);
        this->m_wrap_error =
            std::max(this->m_wrap_error, std::max(tmp_wrap_error_t0_up, tmp_wrap_error_t0_dn));

        NumericalStable::matrix_compare_error(tmp_green_0t_up, this->m_green_0t_up,
                                              tmp_wrap_error_0t_up);
        NumericalStable::matrix_compare_error(tmp_green_0t_dn, this->m_green_0t_dn,
                                              tmp_wrap_error_0t_dn);
        this->m_wrap_error =
            std::max(this->m_wrap_error, std::max(tmp_wrap_error_0t_up, tmp_wrap_error_0t_dn));

        this->m_green_t0_up = tmp_green_t0_up;
        this->m_green_t0_dn = tmp_green_t0_dn;
        this->m_green_0t_up = tmp_green_0t_up;
        this->m_green_0t_dn = tmp_green_0t_dn;

        this->m_vec_green_tt_up[t - 1] = this->m_green_tt_up;
        this->m_vec_green_tt_dn[t - 1] = this->m_green_tt_dn;
        this->m_vec_green_t0_up[t - 1] = this->m_green_t0_up;
        this->m_vec_green_t0_dn[t - 1] = this->m_green_t0_dn;
        this->m_vec_green_0t_up[t - 1] = this->m_green_0t_up;
        this->m_vec_green_0t_dn[t - 1] = this->m_green_0t_dn;

        tmp_mat_up.setIdentity();
        tmp_mat_dn.setIdentity();
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
          << fmt_double("Inverse temperature", this->Beta())
          << fmt_int("Imaginary-time length", this->TimeSize())
          << fmt_double("Imaginary-time interval", this->TimeInterval())
          << fmt_int("Stabilization pace", this->StabilizationPace()) << std::endl;
}

void Walker::output_imaginary_time_grids(std::ostream& ostream) const {
  // output the imaginary-time grids
  ostream << std::format("{:>20d}{:>20.5f}{:>20.5f}\n", this->TimeSize(), this->Beta(),
                         this->TimeInterval());
  for (auto t = 0; t < this->TimeSize(); ++t) {
    ostream << std::format("{:>20d}{:>20.10f}\n", t, (t * this->TimeInterval()));
  }
}

}  // namespace DQMC
