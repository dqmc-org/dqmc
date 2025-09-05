#include "measure/measure_handler.h"

#include <format>

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

void MeasureHandler::initial() {
  Observable::ObservableHandler::initial(this->m_obs_list);

  this->m_is_warmup = (this->m_sweeps_warmup != 0);
  this->m_is_equaltime = (!this->m_eqtime_scalar_obs.empty() ||
                          !this->m_eqtime_vector_obs.empty() || !this->m_eqtime_matrix_obs.empty());
  this->m_is_dynamic = (!this->m_dynamic_scalar_obs.empty() ||
                        !this->m_dynamic_vector_obs.empty() || !this->m_dynamic_matrix_obs.empty());
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
      equaltime_sign->accumulator() /= equaltime_sign->bin_size();

      // normalize observables by the countings and the mean value of the sign
      for (auto& scalar_obs : this->m_eqtime_scalar_obs) {
        if (scalar_obs->name() != "equaltime_sign") {
          scalar_obs->accumulator() /= scalar_obs->bin_size() * equaltime_sign->accumulator();
        }
      }
      for (auto& vector_obs : this->m_eqtime_vector_obs) {
        vector_obs->accumulator() /= vector_obs->bin_size() * equaltime_sign->accumulator();
      }
      for (auto& matrix_obs : this->m_eqtime_matrix_obs) {
        matrix_obs->accumulator() /= matrix_obs->bin_size() * equaltime_sign->accumulator();
      }

      // record the absolute value of sign
      equaltime_sign->accumulator() = std::abs(equaltime_sign->accumulator());
    }
  }

  if (this->m_is_dynamic) {
    if (auto* dynamic_sign = this->find<Observable::Scalar>("dynamic_sign")) {
      // normalize the sign measurment first
      dynamic_sign->accumulator() /= dynamic_sign->bin_size();

      for (auto& scalar_obs : this->m_dynamic_scalar_obs) {
        if (scalar_obs->name() != "dynamic_sign") {
          scalar_obs->accumulator() /= scalar_obs->bin_size() * dynamic_sign->accumulator();
        }
      }
      for (auto& vector_obs : this->m_dynamic_vector_obs) {
        vector_obs->accumulator() /= vector_obs->bin_size() * dynamic_sign->accumulator();
      }
      for (auto& matrix_obs : this->m_dynamic_matrix_obs) {
        matrix_obs->accumulator() /= matrix_obs->bin_size() * dynamic_sign->accumulator();
      }

      // record the absolute value of sign
      dynamic_sign->accumulator() = std::abs(dynamic_sign->accumulator());
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
