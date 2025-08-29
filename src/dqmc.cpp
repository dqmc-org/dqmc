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
#include "utils/progressbar.hpp"
#include "walker.h"

namespace DQMC {

Dqmc::Dqmc(const Config& config) : m_rng(42 + config.seed), m_seed(config.seed) {
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
  m_handler = std::make_unique<Measure::MeasureHandler>(
      config.sweeps_warmup, config.bin_num, config.bin_size, config.sweeps_between_bins,
      config.observables, momentum_idx, momentum_list_indices);
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
    std::cout << ">> Configurations of the bosonic fields set to random.\n";
  } else {
    std::ifstream infile(config.fields_file);
    if (!infile.is_open()) {
      throw std::runtime_error("Dqmc::Dqmc(): failed to open file");
    }
    m_model->read_auxiliary_field_from_stream(infile);
    std::cout << ">> Configurations of the bosonic fields read from the input config file.\n";
  }

  // 10. Final DQMC preparations
  m_walker->initial_svd_stacks(*m_lattice, *m_model);
  m_walker->initial_greens_functions();
  m_walker->initial_config_sign();
}

Dqmc::~Dqmc() = default;

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
    // create progress bar
    ProgressBar progressbar(m_handler->warm_up_sweeps() / 2, m_progress_bar_width,
                            m_progress_bar_complete_char, m_progress_bar_incomplete_char);

    // warm-up sweeps
    for (auto sweep = 1; sweep <= m_handler->warm_up_sweeps() / 2; ++sweep) {
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
  if (m_handler->is_equaltime() || m_handler->is_dynamic()) {
    // create progress bar
    ProgressBar progressbar(m_handler->bins_num() * m_handler->bins_size() / 2,
                            m_progress_bar_width, m_progress_bar_complete_char,
                            m_progress_bar_incomplete_char);

    // measuring sweeps
    for (auto bin = 0; bin < m_handler->bins_num(); ++bin) {
      for (auto sweep = 1; sweep <= m_handler->bins_size() / 2; ++sweep) {
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
      for (auto sweep = 0; sweep < m_handler->sweep_between_bins() / 2; ++sweep) {
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

void Dqmc::info_message(std::ostream& ostream) const {
  if (!ostream) {
    throw std::runtime_error(dqmc_format_error("output stream is not valid."));
  }

  auto duration = this->timer_as_duration();

  auto d = std::chrono::duration_cast<std::chrono::days>(duration);
  duration -= d;

  auto h = std::chrono::duration_cast<std::chrono::hours>(duration);
  duration -= h;

  auto m = std::chrono::duration_cast<std::chrono::minutes>(duration);
  duration -= m;

  auto s = std::chrono::duration_cast<std::chrono::seconds>(duration);
  duration -= s;

  ostream << std::format("\n>> The simulation finished in {}d {}h {}m {}s {}ms.\n", d.count(),
                         h.count(), m.count(), s.count(), duration.count());

  ostream << std::format(">> Maximum of the wrapping error: {:.5e}\n", m_walker->wrap_error());
}

void Dqmc::output_results(std::ostream& ostream) const {
  for (const auto& obs_name : m_handler->observables_list()) {
    if (auto obs = m_handler->find<Observable::Scalar>(obs_name)) {
      Observable::output_observable_to_console(ostream, *obs);
    }
  }
}
}  // namespace DQMC
