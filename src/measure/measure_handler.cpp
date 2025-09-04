#include "measure/measure_handler.h"

#include <format>

#include "lattice/lattice_base.h"
#include "model/model_base.h"
#include "utils/assert.h"
#include "walker.h"

namespace Measure {

MeasureHandler::MeasureHandler(int sweeps_warmup, const std::vector<std::string>& observables,
                               int measured_momentum_idx,
                               const std::vector<int>& measured_momentum_list) {
  set_measure_params(sweeps_warmup);
  set_observables(observables);
  set_measured_momentum(measured_momentum_idx);
  set_measured_momentum_list(measured_momentum_list);
}

void MeasureHandler::set_measure_params(int sweeps_warmup) {
  DQMC_ASSERT(sweeps_warmup >= 0);
  this->m_sweeps_warmup = sweeps_warmup;
}

void MeasureHandler::set_observables(const std::vector<std::string>& obs_list) {
  this->m_obs_list = obs_list;
}

void MeasureHandler::set_measured_momentum(int momentum_index) {
  this->m_momentum = momentum_index;
}

void MeasureHandler::set_measured_momentum_list(const std::vector<int>& momentum_index_list) {
  this->m_momentum_list = momentum_index_list;
}

void MeasureHandler::initial(const LatticeBase& lattice, int time_size) {
  // initialize ObservableHandler
  Observable::ObservableHandler::initial(this->m_obs_list);

  this->m_is_warmup = (this->m_sweeps_warmup != 0);
  this->m_is_equaltime = (!this->m_eqtime_scalar_obs.empty() ||
                          !this->m_eqtime_vector_obs.empty() || !this->m_eqtime_matrix_obs.empty());
  this->m_is_dynamic = (!this->m_dynamic_scalar_obs.empty() ||
                        !this->m_dynamic_vector_obs.empty() || !this->m_dynamic_matrix_obs.empty());

  // set up parameters for the observables
  // for equal-time observables
  if (this->m_is_equaltime) {
    for (auto& scalar_obs : this->m_eqtime_scalar_obs) {
      scalar_obs->set_zero_element(0.0);
      scalar_obs->allocate();
    }
    for (auto& vector_obs : this->m_eqtime_vector_obs) {
      // note that the dimensions of the observable should be adjusted or
      // specialized here
      vector_obs->set_zero_element(Eigen::VectorXd::Zero(time_size));
      vector_obs->allocate();
    }
    for (auto& matrix_obs : this->m_eqtime_matrix_obs) {
      // specialize dimensions for certain observables if needed
      matrix_obs->set_zero_element(
          Eigen::MatrixXd::Zero(lattice.space_size(), lattice.space_size()));
      matrix_obs->allocate();
    }
  }

  // for dynamic observables
  if (this->m_is_dynamic) {
    for (auto& scalar_obs : this->m_dynamic_scalar_obs) {
      scalar_obs->set_zero_element(0.0);
      scalar_obs->allocate();
    }
    for (auto& vector_obs : this->m_dynamic_vector_obs) {
      // specialize dimensions for certain observables if needed
      vector_obs->set_zero_element(Eigen::VectorXd::Zero(time_size));
      vector_obs->allocate();
    }
    for (auto& matrix_obs : this->m_dynamic_matrix_obs) {
      // specialize the dimensions of greens functions
      if (matrix_obs->name() == "greens_functions") {
        // for greens function measure, the rows represent different lattice
        // momentum and the columns represent imaginary-time grids.
        matrix_obs->set_zero_element(
            Eigen::MatrixXd::Zero(this->m_momentum_list.size(), time_size));
        matrix_obs->allocate();
      } else {
        // otherwise initialize by default
        matrix_obs->set_zero_element(
            Eigen::MatrixXd::Zero(lattice.space_size(), lattice.space_size()));
        matrix_obs->allocate();
      }
    }
  }
}

void MeasureHandler::equaltime_measure(const Walker& walker, const ModelBase& model,
                                       const LatticeBase& lattice) {
  for (auto& scalar_obs : this->m_eqtime_scalar_obs) {
    scalar_obs->measure(*this, walker, model, lattice, m_pool);
  }
  for (auto& vector_obs : this->m_eqtime_vector_obs) {
    vector_obs->measure(*this, walker, model, lattice, m_pool);
  }
  for (auto& matrix_obs : this->m_eqtime_matrix_obs) {
    matrix_obs->measure(*this, walker, model, lattice, m_pool);
  }
}

void MeasureHandler::dynamic_measure(const Walker& walker, const ModelBase& model,
                                     const LatticeBase& lattice) {
  for (auto& scalar_obs : this->m_dynamic_scalar_obs) {
    scalar_obs->measure(*this, walker, model, lattice, m_pool);
  }
  for (auto& vector_obs : this->m_dynamic_vector_obs) {
    vector_obs->measure(*this, walker, model, lattice, m_pool);
  }
  for (auto& matrix_obs : this->m_dynamic_matrix_obs) {
    matrix_obs->measure(*this, walker, model, lattice, m_pool);
  }
}

void MeasureHandler::normalize_stats() {
  if (this->m_is_equaltime) {
    if (auto* equaltime_sign = this->find<Observable::Scalar>("equaltime_sign")) {
      // normalize the sign measurement first
      equaltime_sign->tmp_value() /= equaltime_sign->counts();

      // normalize observables by the countings and the mean value of the sign
      for (auto& scalar_obs : this->m_eqtime_scalar_obs) {
        if (scalar_obs->name() != "equaltime_sign") {
          scalar_obs->tmp_value() /= scalar_obs->counts() * equaltime_sign->tmp_value();
        }
      }
      for (auto& vector_obs : this->m_eqtime_vector_obs) {
        vector_obs->tmp_value() /= vector_obs->counts() * equaltime_sign->tmp_value();
      }
      for (auto& matrix_obs : this->m_eqtime_matrix_obs) {
        matrix_obs->tmp_value() /= matrix_obs->counts() * equaltime_sign->tmp_value();
      }

      // record the absolute value of sign
      equaltime_sign->tmp_value() = std::abs(equaltime_sign->tmp_value());
    }
  }

  if (this->m_is_dynamic) {
    if (auto* dynamic_sign = this->find<Observable::Scalar>("dynamic_sign")) {
      // normalize the sign measurment first
      dynamic_sign->tmp_value() /= dynamic_sign->counts();

      for (auto& scalar_obs : this->m_dynamic_scalar_obs) {
        if (scalar_obs->name() != "dynamic_sign") {
          scalar_obs->tmp_value() /= scalar_obs->counts() * dynamic_sign->tmp_value();
        }
      }
      for (auto& vector_obs : this->m_dynamic_vector_obs) {
        vector_obs->tmp_value() /= vector_obs->counts() * dynamic_sign->tmp_value();
      }
      for (auto& matrix_obs : this->m_dynamic_matrix_obs) {
        matrix_obs->tmp_value() /= matrix_obs->counts() * dynamic_sign->tmp_value();
      }

      // record the absolute value of sign
      dynamic_sign->tmp_value() = std::abs(dynamic_sign->tmp_value());
    }
  }
}

void MeasureHandler::start_new_block() {
  if (this->m_is_equaltime) {
    for (auto& scalar_obs : this->m_eqtime_scalar_obs) {
      scalar_obs->start_new_block();
    }
    for (auto& vector_obs : this->m_eqtime_vector_obs) {
      vector_obs->start_new_block();
    }
    for (auto& matrix_obs : this->m_eqtime_matrix_obs) {
      matrix_obs->start_new_block();
    }
  }

  if (this->m_is_dynamic) {
    for (auto& scalar_obs : this->m_dynamic_scalar_obs) {
      scalar_obs->start_new_block();
    }
    for (auto& vector_obs : this->m_dynamic_vector_obs) {
      vector_obs->start_new_block();
    }
    for (auto& matrix_obs : this->m_dynamic_matrix_obs) {
      matrix_obs->start_new_block();
    }
  }
}

void MeasureHandler::finalize_block() {
  if (this->m_is_equaltime) {
    for (auto& scalar_obs : this->m_eqtime_scalar_obs) {
      scalar_obs->finalize_block();
    }
    for (auto& vector_obs : this->m_eqtime_vector_obs) {
      vector_obs->finalize_block();
    }
    for (auto& matrix_obs : this->m_eqtime_matrix_obs) {
      matrix_obs->finalize_block();
    }
  }

  if (this->m_is_dynamic) {
    for (auto& scalar_obs : this->m_dynamic_scalar_obs) {
      scalar_obs->finalize_block();
    }
    for (auto& vector_obs : this->m_dynamic_vector_obs) {
      vector_obs->finalize_block();
    }
    for (auto& matrix_obs : this->m_dynamic_matrix_obs) {
      matrix_obs->finalize_block();
    }
  }
}

double MeasureHandler::get_last_block_avg(const std::string& obs_name) const {
  // Try to find the observable in scalar observables first
  if (auto* scalar_obs = this->find<Observable::Scalar>(obs_name)) {
    return scalar_obs->get_last_block_average();
  }
  throw std::runtime_error("Observable '" + obs_name + "' not found or not a scalar observable");
}

void MeasureHandler::analyse(int optimal_bin_size_blocks) {
  if (this->m_is_equaltime) {
    for (auto& scalar_obs : this->m_eqtime_scalar_obs) {
      scalar_obs->analyse(optimal_bin_size_blocks);
    }
    for (auto& vector_obs : this->m_eqtime_vector_obs) {
      vector_obs->analyse(optimal_bin_size_blocks);
    }
    for (auto& matrix_obs : this->m_eqtime_matrix_obs) {
      matrix_obs->analyse(optimal_bin_size_blocks);
    }
  }

  if (this->m_is_dynamic) {
    for (auto& scalar_obs : this->m_dynamic_scalar_obs) {
      scalar_obs->analyse(optimal_bin_size_blocks);
    }
    for (auto& vector_obs : this->m_dynamic_vector_obs) {
      vector_obs->analyse(optimal_bin_size_blocks);
    }
    for (auto& matrix_obs : this->m_dynamic_matrix_obs) {
      matrix_obs->analyse(optimal_bin_size_blocks);
    }
  }
}

void MeasureHandler::clear_temporary() {
  // clear the temporary data for all the observables
  if (this->m_is_equaltime) {
    for (auto& scalar_obs : this->m_eqtime_scalar_obs) {
      scalar_obs->clear_temporary();
    }
    for (auto& vector_obs : this->m_eqtime_vector_obs) {
      vector_obs->clear_temporary();
    }
    for (auto& matrix_obs : this->m_eqtime_matrix_obs) {
      matrix_obs->clear_temporary();
    }
  }

  if (this->m_is_dynamic) {
    for (auto& scalar_obs : this->m_dynamic_scalar_obs) {
      scalar_obs->clear_temporary();
    }
    for (auto& vector_obs : this->m_dynamic_vector_obs) {
      vector_obs->clear_temporary();
    }
    for (auto& matrix_obs : this->m_dynamic_matrix_obs) {
      matrix_obs->clear_temporary();
    }
  }
}

void MeasureHandler::output_measuring_info(std::ostream& ostream) const {
  auto fmt_str = [](const std::string& desc, const std::string& value) {
    return std::format("{:>30s}{:>7s}{:>24s}\n", desc, "->", value);
  };

  auto fmt_int = [](const std::string& desc, int value) {
    return std::format("{:>30s}{:>7s}{:>24d}\n", desc, "->", value);
  };

  auto bool_to_str = [](bool b) { return b ? "True" : "False"; };

  ostream << "   Measuring Params:\n"
          << fmt_str("Warm up", bool_to_str(this->is_warmup()))
          << fmt_str("Equal-time measure", bool_to_str(this->is_equaltime()))
          << fmt_str("Dynamical measure", bool_to_str(this->is_dynamic())) << std::endl;

  ostream << fmt_int("Sweeps for warmup", this->warm_up_sweeps()) << std::endl;
}

}  // namespace Measure
