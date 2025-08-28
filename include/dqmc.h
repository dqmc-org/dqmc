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
#include <memory>
#include <random>
#include <string>
#include <vector>

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
  int bin_num;
  int bin_size;
  int sweeps_between_bins;

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
  void write_results(const std::string& out_path) const;

  // ---------------------------------------- Useful tools
  void show_progress_bar(bool show);
  void progress_bar_format(unsigned int width, char complete, char incomplete);
  void set_refresh_rate(unsigned int refresh_rate);
  std::chrono::milliseconds timer_as_duration() const;

  // ------------------------------------ Accessors for I/O
  const Model::ModelBase& model() const;
  const Lattice::LatticeBase& lattice() const;
  const Walker& walker() const;
  const Measure::MeasureHandler& handler() const;
  const CheckerBoard::CheckerBoardBase* checkerboard() const;

  // Output
  void initial_message(std::ostream& ostream) const;
  void info_message(std::ostream& ostream) const;
  void output_results(std::ostream& ostream) const;

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

  std::default_random_engine m_rng;
  int m_seed;

  // Progress bar settings
  bool m_show_progress_bar{true};
  unsigned int m_progress_bar_width{70};
  unsigned int m_refresh_rate{10};
  char m_progress_bar_complete_char{'='};
  char m_progress_bar_incomplete_char{' '};

  // Timer
  std::chrono::steady_clock::time_point m_begin_time, m_end_time;
};
}  // namespace DQMC
