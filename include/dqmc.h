#pragma once

/**
 *  This header file defines the DQMC::Dqmc class, which encapsulates
 *  a complete Determinantal Quantum Monte Carlo simulation run.
 *  It holds the simulation context (model, lattice, walker, etc.)
 *  and provides methods to run the simulation's phases, such as
 *  thermalization, measurement, and analysis. It also includes
 *  utilities like a timer and a progress bar for monitoring the simulation.
 */

#include <chrono>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "measure/binning_analyzer.h"
#include "utils/logger.h"
#include "walker.h"

namespace Model {
class ModelBase;
}
namespace Lattice {
class LatticeBase;
}
namespace Measure {
class MeasureHandler;
}
namespace CheckerBoard {
class CheckerBoardBase;
}

namespace DQMC {

struct Config {
  // General
  int seed;
  std::string fields_file;

  // Model
  std::string model_type;
  double hopping_t;
  double onsite_u;
  double chemical_potential;

  // Lattice
  std::string lattice_type;
  std::vector<int> lattice_size;

  // Checkerboard
  bool enable_checkerboard;

  // Monte carlo
  double beta;
  double time_size;
  int stabilization_pace;

  // Measure
  int sweeps_warmup;
  std::string autobinning_target_observable;
  double autobinning_target_rel_error;
  int autobinning_max_sweeps;
  int autobinning_min_sweeps;
  int block_size;

  // Observables
  std::vector<std::string> observables;

  // Momentum parameters
  std::string momentum;
  std::string momentum_list;
};

class Dqmc {
 public:
  explicit Dqmc(const Config& config);

  ~Dqmc();

  void run();
  void write_results(const std::filesystem::path& results_path,
                     const std::filesystem::path& bins_path) const;

  // ---------------------------------------- Useful tools
  std::chrono::milliseconds timer_as_duration() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(m_end_time - m_begin_time);
  }

  // ------------------------------------ Accessors for I/O
  const Model::ModelBase& model() const { return *m_model; }
  const Lattice::LatticeBase& lattice() const { return *m_lattice; }
  const Walker& walker() const { return *m_walker; }
  const Measure::MeasureHandler& handler() const { return *m_handler; }
  const CheckerBoard::CheckerBoardBase* checkerboard() const { return m_checkerboard.get(); }

  // Output
  void initial_message() const;
  void info_message() const;
  void output_results() const;

 private:
  // ------------------------------------ Private subroutines
  // Core simulation phases, now private
  void thermalize();
  void measure();
  void analyse();
  void sweep_forth_and_back();

  // Initialization logic moved from the Initializer class
  void create_modules(const Config& config);
  void initialize_modules();
  void initialize_dqmc();

  // ------------------------------------ Member Variables
  std::unique_ptr<Model::ModelBase> m_model;
  std::unique_ptr<Lattice::LatticeBase> m_lattice;
  std::unique_ptr<Walker> m_walker;
  std::unique_ptr<Measure::MeasureHandler> m_handler;
  std::unique_ptr<CheckerBoard::CheckerBoardBase> m_checkerboard;
  std::unique_ptr<Measure::BinningAnalyzer> m_binning_analyzer;

  const Config& m_config;  // Store config for easy access
  std::default_random_engine m_rng;
  int m_seed;

  // Timer
  std::chrono::steady_clock::time_point m_begin_time, m_end_time;

  // Logger reference
  Utils::Logger& m_logger;
};
}  // namespace DQMC
