#include "model/attractive_hubbard.h"

#include <Eigen/Core>
#include <format>
#include <random>
#include <sstream>
#include <string>
#include <unsupported/Eigen/MatrixFunctions>

#include "lattice/lattice_base.h"
#include "utils/assert.h"
#include "walker.h"

namespace Model {

using RealScalar = double;
using SpaceTimeMat = Eigen::MatrixXd;
using SpaceSpaceMat = Eigen::MatrixXd;

AttractiveHubbard::AttractiveHubbard(RealScalar hopping_t, RealScalar onsite_u,
                                     RealScalar chemical_potential) {
  DQMC_ASSERT(hopping_t >= 0.0);
  DQMC_ASSERT(onsite_u >= 0.0);  // abs of the onsite attractive interaction
  this->m_hopping_t = hopping_t;
  this->m_onsite_u = onsite_u;
  this->m_chemical_potential = chemical_potential;
}

RealScalar AttractiveHubbard::HoppingT() const { return this->m_hopping_t; }

RealScalar AttractiveHubbard::ChemicalPotential() const { return this->m_chemical_potential; }

RealScalar AttractiveHubbard::OnSiteU() const { return this->m_onsite_u; }

void AttractiveHubbard::output_model_info(std::ostream& ostream) const {
  auto fmt_double = [](const std::string& desc, double value) {
    return std::format("{:>30s}{:>7s}{:>24.3f}\n", desc, "->", value);
  };

  ostream << "   Model: Attractive Hubbard\n"
          << fmt_double("Hopping constant 't'", HoppingT())
          << fmt_double("Onsite interaction 'U'", OnSiteU())
          << fmt_double("Checimcal potential 'mu'", ChemicalPotential()) << std::endl;
}

void AttractiveHubbard::output_configuration(std::ostream& ostream) const {
  const int time_size = this->m_bosonic_field.rows();
  const int space_size = this->m_bosonic_field.cols();

  ostream << std::format("{:>20d}{:>20d}\n", time_size, space_size);
  for (auto t = 0; t < time_size; ++t) {
    for (auto i = 0; i < space_size; ++i) {
      ostream << std::format("{:>20d}{:>20d}{:>20.1f}\n", t, i, this->m_bosonic_field(t, i));
    }
  }
}

void AttractiveHubbard::set_model_params(RealScalar hopping_t, RealScalar onsite_u,
                                         RealScalar chemical_potential) {
  DQMC_ASSERT(hopping_t >= 0.0);
  DQMC_ASSERT(onsite_u >= 0.0);  // abs of the onsite attractive interaction
  this->m_hopping_t = hopping_t;
  this->m_onsite_u = onsite_u;
  this->m_chemical_potential = chemical_potential;
}

void AttractiveHubbard::initial_params(const LatticeBase& lattice, const Walker& walker) {
  this->m_space_size = lattice.space_size();
  this->m_time_size = walker.time_size();
  const RealScalar time_interval = walker.time_interval();

  this->m_alpha = std::acosh(std::exp(0.5 * time_interval * this->m_onsite_u));

  const double exp_val_plus = std::exp(-2 * this->m_alpha);
  const double exp_val_minus = std::exp(+2 * this->m_alpha);

  this->m_exp_val_half_diff = (exp_val_plus - exp_val_minus) * 0.5;
  this->m_exp_val_avg = (exp_val_plus + exp_val_minus) * 0.5;

  // allocate memory for bosonic fields
  this->m_bosonic_field.resize(this->m_time_size, this->m_space_size);

  // resize the pre-allocated buffers
  m_exp_V_col_buffer.resize(this->m_space_size);
  m_exp_V_row_buffer.resize(this->m_space_size);
}

void AttractiveHubbard::initial_KV_matrices(const LatticeBase& lattice, const Walker& walker) {
  const int space_size = lattice.space_size();
  const RealScalar time_interval = walker.time_interval();
  const SpaceSpaceMat chemical_potential_mat =
      this->m_chemical_potential * SpaceSpaceMat::Identity(space_size, space_size);
  const SpaceSpaceMat Kmat = -this->m_hopping_t * lattice.hopping_matrix() + chemical_potential_mat;

  this->m_temp_buffer = Matrix(space_size, space_size);
  this->m_expK_mat = (-time_interval * Kmat).exp();
  this->m_inv_expK_mat = (+time_interval * Kmat).exp();

  // in general K matrix is symmetrical
  this->m_trans_expK_mat = this->m_expK_mat.transpose();

  // since V is diagonalized in the Hubbard model
  // there is no need to explicitly compute expV
}

void AttractiveHubbard::initial(const LatticeBase& lattice, const Walker& walker) {
  EigenMallocGuard<true> alloc_guard;

  // initialize model params and allocate memory for bosonic fields
  this->initial_params(lattice, walker);

  // initialize K matrices
  // no need to initialize V matrices because in our model V is diagonalized.
  this->initial_KV_matrices(lattice, walker);
}

void AttractiveHubbard::set_bosonic_fields_to_random(std::default_random_engine& rng) {
  // set configurations of the bosonic fields to random
  const auto time_size = this->m_bosonic_field.rows();
  const auto space_size = this->m_bosonic_field.cols();

  std::bernoulli_distribution bernoulli_dist(0.5);
  for (auto t = 0; t < time_size; ++t) {
    for (auto i = 0; i < space_size; ++i) {
      // for Z2 bosonic field, simply set 1.0 or -1.0
      this->m_bosonic_field(t, i) = bernoulli_dist(rng) ? +1.0 : -1.0;
    }
  }
}

void AttractiveHubbard::update_bosonic_field(TimeIndex time_index, SpaceIndex space_index) {
  DQMC_ASSERT(time_index >= 0 && time_index < this->m_time_size);
  DQMC_ASSERT(space_index >= 0 && space_index < this->m_space_size);

  // for Z2 bosonic fields, a local update is presented by a local Z2 flip
  this->m_bosonic_field(time_index, space_index) = -this->m_bosonic_field(time_index, space_index);
}

double AttractiveHubbard::get_update_ratio(const Walker& walker, TimeIndex time_index,
                                           SpaceIndex space_index) const {
  DQMC_ASSERT(time_index >= 0 && time_index < this->m_time_size);
  DQMC_ASSERT(space_index >= 0 && space_index < this->m_space_size);

  const Eigen::MatrixXd& green_tt_up = walker.green_tt_up();
  const Eigen::MatrixXd& green_tt_dn = walker.green_tt_down();

  const double s = this->m_bosonic_field(time_index, space_index);
  const double exp_factor = this->m_exp_val_half_diff * s + this->m_exp_val_avg;
  const double delta = exp_factor - 1;

  const double det_ratio_up = 1 + (1 - green_tt_up(space_index, space_index)) * delta;
  const double det_ratio_dn = 1 + (1 - green_tt_dn(space_index, space_index)) * delta;

  return det_ratio_up * det_ratio_dn / exp_factor;
}

void AttractiveHubbard::update_greens_function(Walker& walker, TimeIndex time_index,
                                               SpaceIndex space_index) {
  // update the equal-time greens functions
  // as a consequence of a local Z2 flip of the bosonic fields at (time_index,
  // space_index)
  DQMC_ASSERT(time_index >= 0 && time_index < this->m_time_size);
  DQMC_ASSERT(space_index >= 0 && space_index < this->m_space_size);

  Eigen::MatrixXd& green_tt_up = walker.green_tt_up();
  Eigen::MatrixXd& green_tt_dn = walker.green_tt_down();

  // reference:
  //   Quantum Monte Carlo Methods (Algorithms for Lattice Models) Determinant
  //   method
  // here we use the sparseness of the matrix \delta
  const double s = this->m_bosonic_field(time_index, space_index);
  const double exp_factor = this->m_exp_val_half_diff * s + this->m_exp_val_avg;
  const double delta = exp_factor - 1;

  const double factor_up = delta / (1 + (1 - green_tt_up(space_index, space_index)) * delta);

  // for attractive hubbard model, because the spin-up and spin-down parts are
  // coupled to the bosonic fields in the same way, the model possesses the spin
  // degeneracy, and in princile it's sufficient to only simulate one specific
  // spin state.
  const double factor_dn = factor_up;

  const auto unit = Eigen::VectorXd::Unit(this->m_space_size, space_index);
  green_tt_up.noalias() -=
      factor_up * green_tt_up.col(space_index) * (unit.transpose() - green_tt_up.row(space_index));
  green_tt_dn.noalias() -=
      factor_dn * green_tt_dn.col(space_index) * (unit.transpose() - green_tt_dn.row(space_index));
}

void AttractiveHubbard::mult_B_from_left(GreensFunc& green, TimeIndex time_index, Spin spin) const {
  // Multiply a dense matrix, specifically a greens function, from the left by
  // B(t)
  //      G  ->  B(t) * G = exp( -dt V_sigma(t) ) * exp( -dt K ) * G
  // Matrix G is changed in place.
  DQMC_ASSERT(green.rows() == this->m_space_size && green.cols() == this->m_space_size);
  DQMC_ASSERT(time_index >= 0 && time_index <= this->m_time_size);
  // 1.0 for spin up and -1.0 for spin down
  DQMC_ASSERT(abs(spin) == 1.0);

  // due to the periodical boundary condition (PBC)
  // the time slice labeled by 0 actually corresponds to slice tau = beta
  const int eff_time_index = (time_index == 0) ? this->m_time_size - 1 : time_index - 1;
  this->call_mult_expK_from_left(green);
  m_exp_V_col_buffer =
      (this->m_alpha * this->m_bosonic_field.row(eff_time_index).transpose().array()).exp();
  green.array().colwise() *= m_exp_V_col_buffer.array();
}

void AttractiveHubbard::mult_B_from_right(GreensFunc& green, TimeIndex time_index,
                                          Spin spin) const {
  // Multiply a dense matrix, specifically a greens function, from the right by
  // B(t)
  //      G  ->  G * B(t) = G * exp( -dt V_sigma(t) ) * exp( -dt K )
  // Matrix G is changed in place.
  DQMC_ASSERT(green.rows() == this->m_space_size && green.cols() == this->m_space_size);
  DQMC_ASSERT(time_index >= 0 && time_index <= this->m_time_size);
  DQMC_ASSERT(abs(spin) == 1.0);

  const int eff_time_index = (time_index == 0) ? this->m_time_size - 1 : time_index - 1;
  m_exp_V_row_buffer = (this->m_alpha * this->m_bosonic_field.row(eff_time_index).array()).exp();
  green.array().rowwise() *= m_exp_V_row_buffer.array();
  this->call_mult_expK_from_right(green);
}

void AttractiveHubbard::mult_invB_from_left(GreensFunc& green, TimeIndex time_index,
                                            Spin spin) const {
  // Multiply a dense matrix, specifically a greens function, from the left by
  // B(t)^-1
  //      G  ->  B(t)^-1 * G = exp( +dt K ) * exp( +dt V_sigma(t) ) * G
  // Matrix G is changed in place.
  DQMC_ASSERT(green.rows() == this->m_space_size && green.cols() == this->m_space_size);
  DQMC_ASSERT(time_index >= 0 && time_index <= this->m_time_size);
  DQMC_ASSERT(abs(spin) == 1.0);

  const int eff_time_index = (time_index == 0) ? this->m_time_size - 1 : time_index - 1;
  m_exp_V_col_buffer =
      (-this->m_alpha * this->m_bosonic_field.row(eff_time_index).transpose().array()).exp();
  green.array().colwise() *= m_exp_V_col_buffer.array();
  this->call_mult_inv_expK_from_left(green);
}

void AttractiveHubbard::mult_invB_from_right(GreensFunc& green, TimeIndex time_index,
                                             Spin spin) const {
  // Multiply a dense matrix, specifically a greens function, from the right by
  // B(t)^-1
  //      G  ->  G * B(t)^-1 = G * exp( +dt K ) * exp( +dt V_sigma(t) )
  // Matrix G is changed in place.
  DQMC_ASSERT(green.rows() == this->m_space_size && green.cols() == this->m_space_size);
  DQMC_ASSERT(time_index >= 0 && time_index <= this->m_time_size);
  DQMC_ASSERT(abs(spin) == 1.0);

  const int eff_time_index = (time_index == 0) ? this->m_time_size - 1 : time_index - 1;
  this->call_mult_inv_expK_from_right(green);
  m_exp_V_row_buffer = (-this->m_alpha * this->m_bosonic_field.row(eff_time_index).array()).exp();
  green.array().rowwise() *= m_exp_V_row_buffer.array();
}

void AttractiveHubbard::mult_transB_from_left(GreensFunc& green, TimeIndex time_index,
                                              Spin spin) const {
  // Multiply a dense matrix, specifically a greens function, from the left by
  // B(t)^T
  //      G  ->  B(t)^T * G = exp( -dt K )^T * exp( -dt V_sigma(t) ) * G
  // Matrix G is changed in place.
  DQMC_ASSERT(green.rows() == this->m_space_size && green.cols() == this->m_space_size);
  DQMC_ASSERT(time_index >= 0 && time_index <= this->m_time_size);
  DQMC_ASSERT(abs(spin) == 1.0);

  const int eff_time_index = (time_index == 0) ? this->m_time_size - 1 : time_index - 1;
  m_exp_V_col_buffer =
      (this->m_alpha * this->m_bosonic_field.row(eff_time_index).transpose().array()).exp();
  green.array().colwise() *= m_exp_V_col_buffer.array();
  this->call_mult_trans_expK_from_left(green);
}

void AttractiveHubbard::read_auxiliary_field_from_stream(std::istream& infile) {
  std::string line;
  std::string token;
  std::vector<std::string> data;

  std::getline(infile, line);
  std::istringstream iss(line);
  data.clear();
  while (iss >> token) {
    data.push_back(token);
  }

  const int time_size = std::stoi(data[0]);
  const int space_size = std::stoi(data[1]);
  if ((time_size != this->m_bosonic_field.rows()) || (space_size != this->m_bosonic_field.cols())) {
    throw std::runtime_error(
        "AttractiveHubbard::read_auxiliary_field_from_stream(): "
        "inconsistency between model settings and input configs "
        "(time or space size).");
  }

  int time_point, space_point;
  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    data.clear();
    while (iss >> token) {
      data.push_back(token);
    }
    time_point = std::stoi(data[0]);
    space_point = std::stoi(data[1]);
    this->m_bosonic_field(time_point, space_point) = std::stod(data[2]);
  }
}

}  // namespace Model
