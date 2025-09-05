#include "dqmc.h"

#include <algorithm>
#include <format>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "checkerboard/square.h"
#include "lattice/chain.h"
#include "lattice/cubic.h"
#include "lattice/square.h"
#include "measure/measure_handler.h"
#include "measure/observable.h"
#include "model/attractive_hubbard.h"
#include "model/repulsive_hubbard.h"
#include "utils/assert.h"
#include "utils/eigen_malloc_guard.h"
#include "utils/logger.h"
#include "utils/spinner.h"
#include "walker.h"

namespace DQMC {

Dqmc::Dqmc(const Config& config)
    : m_config(config),
      m_rng(42 + config.seed),
      m_seed(config.seed),
      m_logger(Utils::Logger::the()) {
  // 1. Create Lattice
  if (config.lattice_type == "Square") {
    DQMC_ASSERT(config.lattice_size.size() == 2);
    m_lattice = std::make_unique<Lattice::Square>(config.lattice_size);
  } else if (config.lattice_type == "Cubic") {
    DQMC_ASSERT(config.lattice_size.size() == 3);
    m_lattice = std::make_unique<Lattice::Cubic>(config.lattice_size);
  } else if (config.lattice_type == "Chain") {
    DQMC_ASSERT(config.lattice_size.size() == 1);
    m_lattice = std::make_unique<Lattice::Chain>(config.lattice_size);
  } else {
    throw std::runtime_error(
        dqmc_format_error("unsupported lattice type: {}", config.lattice_type));
  }

  // 2. Validate observables against lattice type
  if (config.lattice_type != "Square") {
    if (std::find(config.observables.begin(), config.observables.end(), "superfluid_stiffness") !=
        config.observables.end()) {
      throw std::runtime_error("superfluid_stiffness is only supported for Square lattice");
    }
  }

  // 3. Determine Momentum Indices
  int momentum_idx;
  std::vector<int> momentum_list_indices;
  try {
    momentum_idx = m_lattice->momentum_points().at(config.momentum);
    momentum_list_indices = m_lattice->momentum_lists().at(config.momentum_list);
  } catch (const std::out_of_range& e) {
    throw std::runtime_error(
        dqmc_format_error("unknown momentum point '{}' or list '{}' for {} lattice",
                          config.momentum, config.momentum_list, config.lattice_type));
  }

  // 4. Create MeasureHandler
  m_handler = std::make_unique<Measure::MeasureHandler>(config.sweeps_warmup, config.observables,
                                                        momentum_idx, momentum_list_indices);
  m_handler->initial(*m_lattice, config.time_size);

  // 5. Create Walker
  m_walker = std::make_unique<Walker>(config.beta, config.time_size, config.stabilization_pace);
  m_walker->initial(*m_lattice, *m_handler);

  // 6. Create Model
  if (config.model_type == "RepulsiveHubbard") {
    m_model = std::make_unique<Model::RepulsiveHubbard>(config.hopping_t, config.onsite_u,
                                                        config.chemical_potential);
    m_model->initial(*m_lattice, *m_walker);
  } else if (config.model_type == "AttractiveHubbard") {
    m_model = std::make_unique<Model::AttractiveHubbard>(config.hopping_t, config.onsite_u,
                                                         config.chemical_potential);
    m_model->initial(*m_lattice, *m_walker);
  } else {
    throw std::runtime_error(dqmc_format_error("undefined model type: {}", config.model_type));
  }

  // 7. Create CheckerBoard if enabled
  if (config.enable_checkerboard) {
    if (config.lattice_type == "Square") {
      m_checkerboard = std::make_unique<CheckerBoard::Square>(*m_lattice, *m_model, *m_walker);
    } else {
      throw std::runtime_error(
          dqmc_format_error("checkerboard is currently only implemented for 2d square lattice"));
    }
  }

  // 8. Link checkerboard to the model if it exists
  if (m_checkerboard) {
    m_model->link(*m_checkerboard);
  } else {
    m_model->link();
  }

  // 9. Initialize auxiliary fields
  if (config.fields_file.empty()) {
    m_model->set_bosonic_fields_to_random(m_rng);
    m_logger.info("Configurations of the bosonic fields set to random.");
  } else {
    std::ifstream infile(config.fields_file);
    if (!infile.is_open()) {
      throw std::runtime_error("Dqmc::Dqmc(): failed to open file");
    }
    m_model->read_auxiliary_field_from_stream(infile);
    m_logger.info("Configurations of the bosonic fields read from the input config file.");
  }

  // 10. Final DQMC preparations
  m_walker->initial_svd_stacks(*m_lattice, *m_model);
  m_walker->initial_greens_functions();
  m_walker->initial_config_sign();
}

Dqmc::~Dqmc() = default;

void Dqmc::run() {
  EigenMallocGuard<false> no_alloc_guard;
  m_begin_time = std::chrono::steady_clock::now();
  thermalize();
  measure();
  analyse();
  m_end_time = std::chrono::steady_clock::now();
}

void Dqmc::write_results(const std::string& out_path) const {
  std::ofstream outfile;

  // Output bosonic fields
  outfile.open(std::format("{}/bosonic_fields_{}.out", out_path, m_seed));
  if (!outfile.is_open()) {
    throw std::runtime_error(dqmc_format_error("failed to open file"));
  }
  m_model->output_configuration(outfile);
  outfile.close();

  // Output k stars
  outfile.open(std::format("{}/kstars.out", out_path));
  if (!outfile.is_open()) {
    throw std::runtime_error(dqmc_format_error("failed to open file"));
  }
  m_lattice->output_k_points(outfile);
  outfile.close();

  // Output imaginary-time grids
  outfile.open(std::format("{}/imaginary_time_grids.out", out_path));
  if (!outfile.is_open()) {
    throw std::runtime_error(dqmc_format_error("failed to open file"));
  }
  m_walker->output_imaginary_time_grids(outfile);
  outfile.close();

  // Helper lambda for observables
  auto output_observable_files = [&](const auto& obs, const std::string& obs_name) {
    outfile.open(std::format("{}/{}_{}.out", out_path, obs_name, m_seed));
    Observable::output_observable_to_file(outfile, *obs);
    outfile.close();

    outfile.open(std::format("{}/{}.bins.out", out_path, obs_name));
    Observable::output_observable_in_bins_to_file(outfile, *obs);
    outfile.close();
  };

  // Iterate and output all observables
  for (const auto& obs_name : handler().observables_list()) {
    if (auto obs = handler().find<Observable::Scalar>(obs_name)) {
      output_observable_files(obs, obs_name);
    } else if (auto obs = handler().find<Observable::Vector>(obs_name)) {
      output_observable_files(obs, obs_name);
    } else if (auto obs = handler().find<Observable::Matrix>(obs_name)) {
      output_observable_files(obs, obs_name);
    }
  }
}

// -----------------------------------  Implementation of private methods
// --------------------------------------

void Dqmc::sweep_forth_and_back() {
  if (m_handler->is_dynamic()) {
    m_walker->sweep_for_dynamic_greens(*m_model);
    m_handler->dynamic_measure(*m_walker, *m_model, *m_lattice);
  } else {
    m_walker->sweep_from_0_to_beta(*m_model, m_rng);
    if (m_handler->is_equaltime()) {
      m_handler->equaltime_measure(*m_walker, *m_model, *m_lattice);
    }
  }
  m_walker->sweep_from_beta_to_0(*m_model, m_rng);
  if (m_handler->is_equaltime()) {
    m_handler->equaltime_measure(*m_walker, *m_model, *m_lattice);
  }
}

void Dqmc::thermalize() {
  if (m_handler->is_warmup()) {
    m_logger.info("Warming up...");
    Utils::Spinner spinner{};

    // warm-up sweeps
    for (auto sweep = 1; sweep <= m_handler->warm_up_sweeps() / 2; ++sweep) {
      // sweep forth and back without measuring
      m_walker->sweep_from_0_to_beta(*m_model, m_rng);
      m_walker->sweep_from_beta_to_0(*m_model, m_rng);
      spinner.spin();
    }
  }
}

void Dqmc::measure() {
  if (!m_handler->is_equaltime() && !m_handler->is_dynamic()) return;

  m_binning_analyzer = std::make_unique<Measure::BinningAnalyzer>();
  int total_sweeps = 0;
  int sweeps_in_current_block = 0;
  const int min_sweeps = m_config.autobinning_min_sweeps;
  const int block_target_count = min_sweeps / m_config.block_size / 2;

  m_logger.info("Measuring...");
  Utils::Spinner spinner{};

  m_handler->start_new_block();

  while (total_sweeps < m_config.autobinning_max_sweeps) {
    sweep_forth_and_back();  // This now accumulates into current block
    total_sweeps += 2;
    sweeps_in_current_block += 2;

    if (sweeps_in_current_block >= m_config.block_size) {
      m_handler->normalize_stats();  // Normalize block accumulator
      m_handler->finalize_block();   // Push block average to time-series

      double tracked_value = m_handler->get_last_block_avg(m_config.autobinning_target_observable);
      m_binning_analyzer->add_data_point(tracked_value);

      const int num_blocks = m_binning_analyzer->get_num_data_points();
      if (total_sweeps >= min_sweeps && (num_blocks % block_target_count == 0)) {
        m_binning_analyzer->update_analysis();

        m_logger.debug("Binning statistics:");
        m_logger.debug("tracking observable: {}", m_config.autobinning_target_observable);
        m_logger.debug("total_sweeps: {}", total_sweeps);
        m_logger.debug("mean: {}", m_binning_analyzer->get_mean());
        m_logger.debug("error: {}", m_binning_analyzer->get_error());
        m_logger.debug("autocorrelation_time: {}", m_binning_analyzer->get_autocorrelation_time());
        m_logger.debug("optimal_window_size: {}", m_binning_analyzer->get_optimal_window_size());
        m_logger.debug("num_data_points: {}", m_binning_analyzer->get_num_data_points());

        if (m_binning_analyzer->is_converged(m_config.autobinning_target_rel_error)) {
          m_logger.info("Convergence target reached after {} sweeps.", total_sweeps);
          break;
        }
      }

      m_handler->start_new_block();
      sweeps_in_current_block = 0;
    }

    spinner.spin();
  }

  if (total_sweeps >= m_config.autobinning_max_sweeps) {
    m_logger.info("Maximum number of sweeps reached.");
  }
}

void Dqmc::analyse() {
  // Get the optimal window size (in blocks) from the analyzer
  const int optimal_bin_size_blocks = m_binning_analyzer->get_optimal_window_size();

  if (optimal_bin_size_blocks > 0) {
    // Pass this to the handler to perform final analysis on all observables
    m_handler->analyse(optimal_bin_size_blocks);
  } else {
    m_logger.warn("Not enough data for a reliable analysis. Reporting raw statistics from blocks.");
    m_handler->analyse(1);  // Fallback: treat each block as a bin
  }
}

void Dqmc::initial_message(std::ostream& ostream) const {
  if (!ostream) {
    throw std::runtime_error(dqmc_format_error("output stream is not valid."));
  }

  m_model->output_model_info(ostream);
  m_lattice->output_lattice_info(ostream, m_handler->momentum());

  ostream << std::format("{:>30s}{:>7s}{:>24s}\n\n", "Checkerboard breakups", "->",
                         checkerboard() ? "True" : "False");

  m_walker->output_montecarlo_info(ostream);
  m_handler->output_measuring_info(ostream);
}

void Dqmc::info_message() const {
  auto duration = this->timer_as_duration();

  auto d = std::chrono::duration_cast<std::chrono::days>(duration);
  duration -= d;

  auto h = std::chrono::duration_cast<std::chrono::hours>(duration);
  duration -= h;

  auto m = std::chrono::duration_cast<std::chrono::minutes>(duration);
  duration -= m;

  auto s = std::chrono::duration_cast<std::chrono::seconds>(duration);
  duration -= s;

  m_logger.info("The simulation finished in {}d {}h {}m {}s {}ms.", d.count(), h.count(), m.count(),
                s.count(), duration.count());

  m_logger.info("Maximum of the wrapping error: {:.5e}", m_walker->wrap_error());
}

void Dqmc::output_results(std::ostream& ostream) const {
  for (const auto& obs_name : m_handler->observables_list()) {
    if (auto obs = m_handler->find<Observable::Scalar>(obs_name)) {
      Observable::output_observable_to_console(ostream, *obs);
    }
  }
}
}  // namespace DQMC
