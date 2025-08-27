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

Dqmc::Dqmc(const Config& config) : m_rng(config.seed), m_seed(config.seed) {
  // 1. Create all modules from config
  m_context = parse_config(config);

  // 2. Initialize and link modules
  initial_modules(m_context);

  // 3. Initialize auxiliary fields
  if (config.fields_file.empty()) {
    m_context.model->set_bosonic_fields_to_random(m_rng);
    std::cout << ">> Configurations of the bosonic fields set to random.\n" << std::endl;
  } else {
    IO::read_bosonic_fields_from_file(config.fields_file, *m_context.model);
    std::cout << ">> Configurations of the bosonic fields read from the input config file.\n"
              << std::endl;
  }

  // 4. Final DQMC preparations
  initial_dqmc(m_context);

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
const Model::ModelBase& Dqmc::model() const { return *m_context.model; }
const Lattice::LatticeBase& Dqmc::lattice() const { return *m_context.lattice; }
const Walker& Dqmc::walker() const { return *m_context.walker; }
const Measure::MeasureHandler& Dqmc::handler() const { return *m_context.handler; }
const CheckerBoard::CheckerBoardBase* Dqmc::checkerboard() const {
  return m_context.checkerboard.get();
}

// -----------------------------------  Implementation of private methods
// --------------------------------------

void Dqmc::sweep_forth_and_back() {
  if (m_context.handler->isDynamic()) {
    m_context.walker->sweep_for_dynamic_greens(*m_context.model);
    m_context.handler->dynamic_measure(*m_context.walker, *m_context.model, *m_context.lattice);
  } else {
    m_context.walker->sweep_from_0_to_beta(*m_context.model, m_rng);
    if (m_context.handler->isEqualTime()) {
      m_context.handler->equaltime_measure(*m_context.walker, *m_context.model, *m_context.lattice);
    }
  }
  m_context.walker->sweep_from_beta_to_0(*m_context.model, m_rng);
  if (m_context.handler->isEqualTime()) {
    m_context.handler->equaltime_measure(*m_context.walker, *m_context.model, *m_context.lattice);
  }
}

void Dqmc::thermalize() {
  if (m_context.handler->isWarmUp()) {
    // create progress bar
    ProgressBar progressbar(m_context.handler->WarmUpSweeps() / 2, m_progress_bar_width,
                            m_progress_bar_complete_char, m_progress_bar_incomplete_char);

    // warm-up sweeps
    for (auto sweep = 1; sweep <= m_context.handler->WarmUpSweeps() / 2; ++sweep) {
      // sweep forth and back without measuring
      m_context.walker->sweep_from_0_to_beta(*m_context.model, m_rng);
      m_context.walker->sweep_from_beta_to_0(*m_context.model, m_rng);

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
  if (m_context.handler->isEqualTime() || m_context.handler->isDynamic()) {
    // create progress bar
    ProgressBar progressbar(m_context.handler->BinsNum() * m_context.handler->BinsSize() / 2,
                            m_progress_bar_width, m_progress_bar_complete_char,
                            m_progress_bar_incomplete_char);

    // measuring sweeps
    for (auto bin = 0; bin < m_context.handler->BinsNum(); ++bin) {
      for (auto sweep = 1; sweep <= m_context.handler->BinsSize() / 2; ++sweep) {
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
      m_context.handler->normalize_stats();
      m_context.handler->write_stats_to_bins(bin);
      m_context.handler->clear_temporary();

      // avoid correlations between adjoining bins
      for (auto sweep = 0; sweep < m_context.handler->SweepsBetweenBins() / 2; ++sweep) {
        m_context.walker->sweep_from_0_to_beta(*m_context.model, m_rng);
        m_context.walker->sweep_from_beta_to_0(*m_context.model, m_rng);
      }
    }

    // progress bar finish
    if (m_show_progress_bar) {
      std::cout << " Measuring  ";
      progressbar.done();
    }
  }
}

void Dqmc::analyse() { m_context.handler->analyse_stats(); }

Context Dqmc::parse_config(const Config& config) {
  Context context;
  // --------------------------------------------------------------------------------------------------
  //                                      Parse the Model module
  // --------------------------------------------------------------------------------------------------
  if (config.model_type == "RepulsiveHubbard") {
    context.model = std::make_unique<Model::RepulsiveHubbard>();
    context.model->set_model_params(config.hopping_t, config.onsite_u, config.chemical_potential);
  }

  // -----------------------------------  Attractive Hubbard model
  // -----------------------------------
  else if (config.model_type == "AttractiveHubbard") {
    context.model = std::make_unique<Model::AttractiveHubbard>();
    context.model->set_model_params(config.hopping_t, config.onsite_u, config.chemical_potential);
  }

  else {
    throw std::runtime_error(
        "DQMC::Dqmc::parse_config(): "
        "undefined model type");
  }

  // --------------------------------------------------------------------------------------------------
  //                                    Parse the Lattice module
  // --------------------------------------------------------------------------------------------------
  if (config.lattice_type == "Square") {
    DQMC_ASSERT(config.lattice_size.size() == 2);

    // create 2d square lattice object
    context.lattice = std::make_unique<Lattice::Square>();
    context.lattice->set_lattice_params(config.lattice_size);
    context.lattice->initial();
  }

  else if (config.lattice_type == "Cubic") {
    DQMC_ASSERT(config.lattice_size.size() == 3);

    // create 3d cubic lattice object
    context.lattice = std::make_unique<Lattice::Cubic>();
    context.lattice->set_lattice_params(config.lattice_size);
    context.lattice->initial();
  }

  // ------------------------------------  2D Honeycomb lattice
  // --------------------------------------
  else if (config.lattice_type == "Honeycomb") {
    throw std::runtime_error("Honeycomb lattice is not supported.");
  }

  else {
    throw std::runtime_error(
        "DQMC::Dqmc::parse_config(): "
        "unsupported lattice type");
  }

  // --------------------------------------------------------------------------------------------------
  //                                  Parse the CheckerBoard module
  // --------------------------------------------------------------------------------------------------
  // note that the checkerboard method is currently only implemented for 2d
  // square lattice

  if (config.enable_checkerboard) {
    if (config.lattice_type == "Square") {
      context.checkerboard = std::make_unique<CheckerBoard::Square>();
    } else {
      throw std::runtime_error(
          "DQMC::Dqmc::parse_config(): "
          "checkerboard is currently only implemented for "
          "2d square lattice");
    }
  }

  // --------------------------------------------------------------------------------------------------
  //                                   Parse the Walker module
  // --------------------------------------------------------------------------------------------------
  // create dqmc walker and set up parameters
  context.walker = std::make_unique<Walker>();
  context.walker->set_physical_params(config.beta, config.time_size);
  context.walker->set_stabilization_pace(config.stabilization_pace);

  // --------------------------------------------------------------------------------------------------
  //                                Parse the Measure Handler module
  // --------------------------------------------------------------------------------------------------

  // special observables, e.g. superfluid stiffness, are only supported for
  // specific lattice type.
  if (config.lattice_type != "Square") {
    if (std::find(config.observables.begin(), config.observables.end(), "superfluid_stiffness") !=
        config.observables.end()) {
      throw std::runtime_error("superfluid_stiffness is only supported for Square lattice");
    }
  }

  // create measure handler and set up parameters
  context.handler = std::make_unique<Measure::MeasureHandler>();

  context.handler->set_measure_params(config.sweeps_warmup, config.bin_num, config.bin_size,
                                      config.sweeps_between_bins);
  context.handler->set_observables(config.observables);

  // --------------------------------------------------------------------------------------------------
  //                                Parse the input Momentum parmas
  // --------------------------------------------------------------------------------------------------
  // make sure that the lattice module is initialized ahead
  DQMC_ASSERT(context.lattice->InitialStatus());

  if (config.lattice_type == "Square") {
    // covert base class pointer to that of the derived square lattice class
    if (const auto square_lattice = dynamic_cast<const Lattice::Square*>(context.lattice.get())) {
      if (config.momentum == "GammaPoint") {
        context.handler->set_measured_momentum(square_lattice->GammaPointIndex());
      } else if (config.momentum == "MPoint") {
        context.handler->set_measured_momentum(square_lattice->MPointIndex());
      } else if (config.momentum == "XPoint") {
        context.handler->set_measured_momentum(square_lattice->XPointIndex());
      } else {
        throw std::runtime_error(
            std::format("DQMC::Dqmc::parse_config(): undefined momentum '{}' for 2d square lattice",
                        config.momentum));
      }

      if (config.momentum_list == "KstarsAll") {
        context.handler->set_measured_momentum_list(square_lattice->kStarsIndex());
      } else if (config.momentum_list == "DeltaLine") {
        context.handler->set_measured_momentum_list(square_lattice->DeltaLineIndex());
      } else if (config.momentum_list == "ZLine") {
        context.handler->set_measured_momentum_list(square_lattice->ZLineIndex());
      } else if (config.momentum_list == "SigmaLine") {
        context.handler->set_measured_momentum_list(square_lattice->SigmaLineIndex());
      } else if (config.momentum_list == "Gamma2X2M2GammaLoop") {
        context.handler->set_measured_momentum_list(square_lattice->Gamma2X2M2GammaLoopIndex());
      } else {
        throw std::runtime_error(std::format(
            "DQMC::Dqmc::parse_config(): undefined momentum list '{}' for 2d square lattice",
            config.momentum_list));
      }
    } else {
      throw std::runtime_error(
          "DQMC::Dqmc::parse_config(): fail to convert 'Lattice::LatticeBase' to "
          "'Lattice::Square'.");
    }
  }

  if (config.lattice_type == "Cubic") {
    // covert base class pointer to that of the derived cubic lattice class
    if (const auto cubic_lattice = dynamic_cast<const Lattice::Cubic*>(context.lattice.get())) {
      if (config.momentum == "GammaPoint") {
        context.handler->set_measured_momentum(cubic_lattice->GammaPointIndex());
      } else if (config.momentum == "MPoint") {
        context.handler->set_measured_momentum(cubic_lattice->MPointIndex());
      } else if (config.momentum == "XPoint") {
        context.handler->set_measured_momentum(cubic_lattice->XPointIndex());
      } else if (config.momentum == "RPoint") {
        context.handler->set_measured_momentum(cubic_lattice->RPointIndex());
      } else {
        throw std::runtime_error(
            std::format("DQMC::Dqmc::parse_config(): undefined momentum '{}' for 3d cubic lattice",
                        config.momentum));
      }

      if (config.momentum_list == "KstarsAll") {
        context.handler->set_measured_momentum_list(cubic_lattice->kStarsIndex());
      } else if (config.momentum_list == "DeltaLine") {
        context.handler->set_measured_momentum_list(cubic_lattice->DeltaLineIndex());
      } else if (config.momentum_list == "ZLine") {
        context.handler->set_measured_momentum_list(cubic_lattice->ZLineIndex());
      } else if (config.momentum_list == "SigmaLine") {
        context.handler->set_measured_momentum_list(cubic_lattice->SigmaLineIndex());
      } else if (config.momentum_list == "LambdaLine") {
        context.handler->set_measured_momentum_list(cubic_lattice->LambdaLineIndex());
      } else if (config.momentum_list == "SLine") {
        context.handler->set_measured_momentum_list(cubic_lattice->SLineIndex());
      } else if (config.momentum_list == "TLine") {
        context.handler->set_measured_momentum_list(cubic_lattice->TLineIndex());
      } else {
        throw std::runtime_error(std::format(
            "DQMC::Dqmc::parse_config(): undefined momentum list '{}' for 3d cubic lattice",
            config.momentum_list));
      }
    } else {
      throw std::runtime_error(
          "DQMC::Dqmc::parse_config(): fail to convert 'Lattice::LatticeBase' to "
          "'Lattice::Cubic'.");
    }
  }

  if (config.lattice_type == "Honeycomb") {
    throw std::runtime_error("Honeycomb lattice is not supported.");
  }

  return context;
}

void Dqmc::initial_modules(const Context& context) {
  // NOTE: the order of initializations below are important.
  DQMC_ASSERT(context.lattice->InitialStatus());

  context.handler->initial(*context.lattice, *context.walker);
  context.walker->initial(*context.lattice, *context.handler);
  context.model->initial(*context.lattice, *context.walker);

  if (context.checkerboard) {
    context.checkerboard->set_checkerboard_params(*context.lattice, *context.model,
                                                  *context.walker);
    context.checkerboard->initial();
    context.model->link(*context.checkerboard);
  } else {
    context.model->link();
  }
}

void Dqmc::initial_dqmc(const Context& context) {
  // NOTE: this should be called after the initial configuration of the bosonic
  // fields.
  context.walker->initial_svd_stacks(*context.lattice, *context.model);
  context.walker->initial_greens_functions();
  context.walker->initial_config_sign();
}
}  // namespace DQMC
