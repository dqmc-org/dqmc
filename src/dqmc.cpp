#include "dqmc.h"

#include <algorithm>
#include <format>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "checkerboard/cubic.h"
#include "checkerboard/square.h"
#include "io.h"
#include "lattice/cubic.h"
#include "lattice/honeycomb.h"
#include "lattice/square.h"
#include "measure/measure_handler.h"
#include "measure/observable.h"
#include "model/attractive_hubbard.h"
#include "model/repulsive_hubbard.h"
#include "svd_stack.h"
#include "utils/assert.h"
#include "utils/progressbar.hpp"
#include "walker.h"

namespace DQMC {

Dqmc::Dqmc(const Config& config) : m_rng(42 + config.seed), m_seed(config.seed) {
  // 1. Create all modules from config
  create_modules(config);

  // 2. Initialize and link modules
  initialize_modules();

  // 3. Initialize auxiliary fields
  if (config.fields_file.empty()) {
    m_model->set_bosonic_fields_to_random(m_rng);
    std::cout << ">> Configurations of the bosonic fields set to random.\n" << std::endl;
  } else {
    IO::read_bosonic_fields_from_file(config.fields_file, *m_model);
    std::cout << ">> Configurations of the bosonic fields read from the input config file.\n"
              << std::endl;
  }

  // 4. Final DQMC preparations
  initialize_dqmc();

  std::cout << ">> Initialization finished. \n\n"
            << ">> The simulation is going to get started with parameters shown below :\n"
            << std::endl;
}

void Dqmc::run() {
  m_begin_time = std::chrono::steady_clock::now();
  thermalize();
  measure();
  analyse();
  m_end_time = std::chrono::steady_clock::now();
}

void Dqmc::write_results(const std::string& out_path) const {
  std::ofstream outfile;

  // Output bosonic fields
  const auto fields_out = std::format("{}/bosonic_fields_{}.out", out_path, m_seed);
  outfile.open(fields_out, std::ios::trunc);
  IO::output_bosonic_fields(outfile, model());
  outfile.close();

  // Output k stars
  outfile.open(std::format("{}/kstars.out", out_path), std::ios::trunc);
  IO::output_k_stars(outfile, lattice());
  outfile.close();

  // Output imaginary-time grids
  outfile.open(std::format("{}/imaginary_time_grids.out", out_path), std::ios::trunc);
  IO::output_imaginary_time_grids(outfile, walker());
  outfile.close();

  // Helper lambda for observables
  auto output_observable_files = [&](const auto& obs, const std::string_view& obs_name) {
    outfile.open(std::format("{}/{}_{}.out", out_path, obs_name, m_seed), std::ios::trunc);
    IO::output_observable_to_file(outfile, *obs);
    outfile.close();

    outfile.open(std::format("{}/{}.bins.out", out_path, obs_name), std::ios::trunc);
    IO::output_observable_in_bins_to_file(outfile, *obs);
    outfile.close();
  };

  // Iterate and output all observables
  for (const auto& obs_name : handler().ObservablesList()) {
    if (auto obs = handler().find<Observable::Scalar>(obs_name)) {
      output_observable_files(obs, obs_name);
    } else if (auto obs = handler().find<Observable::Vector>(obs_name)) {
      output_observable_files(obs, obs_name);
    } else if (auto obs = handler().find<Observable::Matrix>(obs_name)) {
      output_observable_files(obs, obs_name);
    }
  }
}

// -----------------------------------  Useful tools
// --------------------------------------

// set up whether to show the process bar or not
void Dqmc::show_progress_bar(bool show) { m_show_progress_bar = show; }

// set up the format of the progress bar
void Dqmc::progress_bar_format(unsigned int width, char complete, char incomplete) {
  m_progress_bar_width = width;
  m_progress_bar_complete_char = complete;
  m_progress_bar_incomplete_char = incomplete;
}

// set up the rate of refreshing the progress bar
void Dqmc::set_refresh_rate(unsigned int refresh_rate) { m_refresh_rate = refresh_rate; }

std::chrono::milliseconds Dqmc::timer_as_duration() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(m_end_time - m_begin_time);
}

// ------------------------------------ Accessors for I/O -------------------------------------
const Model::ModelBase& Dqmc::model() const { return *m_model; }
const Lattice::LatticeBase& Dqmc::lattice() const { return *m_lattice; }
const Walker& Dqmc::walker() const { return *m_walker; }
const Measure::MeasureHandler& Dqmc::handler() const { return *m_handler; }
const CheckerBoard::CheckerBoardBase* Dqmc::checkerboard() const { return m_checkerboard.get(); }

// -----------------------------------  Implementation of private methods
// --------------------------------------

void Dqmc::sweep_forth_and_back() {
  if (m_handler->isDynamic()) {
    m_walker->sweep_for_dynamic_greens(*m_model);
    m_handler->dynamic_measure(*m_walker, *m_model, *m_lattice);
  } else {
    m_walker->sweep_from_0_to_beta(*m_model, m_rng);
    if (m_handler->isEqualTime()) {
      m_handler->equaltime_measure(*m_walker, *m_model, *m_lattice);
    }
  }
  m_walker->sweep_from_beta_to_0(*m_model, m_rng);
  if (m_handler->isEqualTime()) {
    m_handler->equaltime_measure(*m_walker, *m_model, *m_lattice);
  }
}

void Dqmc::thermalize() {
  if (m_handler->isWarmUp()) {
    // create progress bar
    ProgressBar progressbar(m_handler->WarmUpSweeps() / 2, m_progress_bar_width,
                            m_progress_bar_complete_char, m_progress_bar_incomplete_char);

    // warm-up sweeps
    for (auto sweep = 1; sweep <= m_handler->WarmUpSweeps() / 2; ++sweep) {
      // sweep forth and back without measuring
      m_walker->sweep_from_0_to_beta(*m_model, m_rng);
      m_walker->sweep_from_beta_to_0(*m_model, m_rng);

      // record the tick
      ++progressbar;
      if (m_show_progress_bar && (sweep % m_refresh_rate == 1)) {
        std::cout << " Warming up ";
        progressbar.display();
      }
    }

    // progress bar finish
    if (m_show_progress_bar) {
      std::cout << " Warming up ";
      progressbar.done();
    }
  }
}

void Dqmc::measure() {
  if (m_handler->isEqualTime() || m_handler->isDynamic()) {
    // create progress bar
    ProgressBar progressbar(m_handler->BinsNum() * m_handler->BinsSize() / 2, m_progress_bar_width,
                            m_progress_bar_complete_char, m_progress_bar_incomplete_char);

    // measuring sweeps
    for (auto bin = 0; bin < m_handler->BinsNum(); ++bin) {
      for (auto sweep = 1; sweep <= m_handler->BinsSize() / 2; ++sweep) {
        // update and measure
        sweep_forth_and_back();

        // record the tick
        ++progressbar;
        if (m_show_progress_bar && (sweep % m_refresh_rate == 1)) {
          std::cout << " Measuring  ";
          progressbar.display();
        }
      }

      // store the collected data in the MeasureHandler
      m_handler->normalize_stats();
      m_handler->write_stats_to_bins(bin);
      m_handler->clear_temporary();

      // avoid correlations between adjoining bins
      for (auto sweep = 0; sweep < m_handler->SweepsBetweenBins() / 2; ++sweep) {
        m_walker->sweep_from_0_to_beta(*m_model, m_rng);
        m_walker->sweep_from_beta_to_0(*m_model, m_rng);
      }
    }

    // progress bar finish
    if (m_show_progress_bar) {
      std::cout << " Measuring  ";
      progressbar.done();
    }
  }
}

void Dqmc::analyse() { m_handler->analyse_stats(); }

void Dqmc::create_modules(const Config& config) {
  // 1. Create Lattice
  if (config.lattice_type == "Square") {
    DQMC_ASSERT(config.lattice_size.size() == 2);
    m_lattice = std::make_unique<Lattice::Square>(config.lattice_size);
  } else if (config.lattice_type == "Cubic") {
    DQMC_ASSERT(config.lattice_size.size() == 3);
    m_lattice = std::make_unique<Lattice::Cubic>(config.lattice_size);
  } else if (config.lattice_type == "Honeycomb") {
    throw std::runtime_error("Honeycomb lattice is not supported.");
  } else {
    throw std::runtime_error("DQMC::Dqmc::create_modules(): unsupported lattice type");
  }

  // 2. Create Model
  if (config.model_type == "RepulsiveHubbard") {
    m_model = std::make_unique<Model::RepulsiveHubbard>(config.hopping_t, config.onsite_u,
                                                        config.chemical_potential);
  } else if (config.model_type == "AttractiveHubbard") {
    m_model = std::make_unique<Model::AttractiveHubbard>(config.hopping_t, config.onsite_u,
                                                         config.chemical_potential);
  } else {
    throw std::runtime_error("DQMC::Dqmc::create_modules(): undefined model type");
  }

  // 3. Create Walker
  m_walker = std::make_unique<Walker>(config.beta, config.time_size, config.stabilization_pace);

  // 4. Determine Momentum Indices for MeasureHandler
  if (config.lattice_type == "Honeycomb") {
    throw std::runtime_error("Honeycomb lattice is not supported.");
  }

  int momentum_idx;
  std::vector<int> momentum_list_indices;
  try {
    momentum_idx = m_lattice->momentum_points().at(config.momentum);
    momentum_list_indices = m_lattice->momentum_lists().at(config.momentum_list);
  } catch (const std::out_of_range& e) {
    throw std::runtime_error(std::format(
        "DQMC::create_modules(): unknown momentum point '{}' or list '{}' for {} lattice",
        config.momentum, config.momentum_list, config.lattice_type));
  }

  // 5. Validate observables against lattice type
  if (config.lattice_type != "Square") {
    if (std::find(config.observables.begin(), config.observables.end(), "superfluid_stiffness") !=
        config.observables.end()) {
      throw std::runtime_error("superfluid_stiffness is only supported for Square lattice");
    }
  }

  // 6. Create MeasureHandler
  m_handler = std::make_unique<Measure::MeasureHandler>(
      config.sweeps_warmup, config.bin_num, config.bin_size, config.sweeps_between_bins,
      config.observables, momentum_idx, momentum_list_indices);

  // 7. Create CheckerBoard if enabled
  if (config.enable_checkerboard) {
    if (config.lattice_type == "Square") {
      m_checkerboard = std::make_unique<CheckerBoard::Square>(*m_lattice, *m_model, *m_walker);
    } else {
      throw std::runtime_error(
          "DQMC::Dqmc::create_modules(): "
          "checkerboard is currently only implemented for 2d square lattice");
    }
  }
}

void Dqmc::initialize_modules() {
  // NOTE: The order of initializations below remains important for inter-linking.
  DQMC_ASSERT(m_lattice->InitialStatus());

  m_handler->initial(*m_lattice, *m_walker);
  m_walker->initial(*m_lattice, *m_handler);
  m_model->initial(*m_lattice, *m_walker);

  // Link checkerboard to the model if it exists
  if (m_checkerboard) {
    m_model->link(*m_checkerboard);
  } else {
    m_model->link();
  }
}

void Dqmc::initialize_dqmc() {
  // NOTE: this should be called after the initial configuration of the bosonic
  // fields.
  m_walker->initial_svd_stacks(*m_lattice, *m_model);
  m_walker->initial_greens_functions();
  m_walker->initial_config_sign();
}
}  // namespace DQMC
