#pragma once

/**
 *  This header file defines the DQMC::Dqmc class, which encapsulates
 *  a complete Determinental Quantum Monte Carlo simulation run.
 *  It holds the simulation context (model, lattice, walker, etc.)
 *  and provides methods to run the simulation's phases, such as
 *  thermalization, measurement, and analysis. It also includes
 *  utilities like a timer and a progress bar for monitoring the simulation.
 */

#include <chrono>
#include <random>

#include "checkerboard/checkerboard_base.h"
#include "initializer.h"
#include "lattice/lattice_base.h"
#include "measure/measure_handler.h"
#include "model/model_base.h"
#include "walker.h"

namespace DQMC {

using ModelBase = Model::ModelBase;
using LatticeBase = Lattice::LatticeBase;
using MeasureHandler = Measure::MeasureHandler;
using CheckerBoardBase = CheckerBoard::CheckerBoardBase;

// --------------------------- The main DQMC simulation class ---------------------------
class Dqmc {
 public:
  // Constructor takes ownership of the simulation context.
  explicit Dqmc(Context&& context);

  // ---------------------------------------- Useful tools
  // ------------------------------------------

  void show_progress_bar(bool show);
  void progress_bar_format(unsigned int width, char complete, char incomplete);
  void set_refresh_rate(unsigned int refresh_rate);

  double timer() const;
  std::chrono::milliseconds timer_as_duration() const;
  void timer_begin();
  void timer_end();

  // ------------------------------------ Crucial Dqmc routines
  // -------------------------------------

  // Thermalization of the field configurations
  void thermalize(std::default_random_engine& rng);

  // Monte Carlo updates and measurements
  void measure(std::default_random_engine& rng);

  // Analyse the measured data
  void analyse();

  // ------------------------------------ Accessors for I/O
  // -------------------------------------
  const ModelBase& model() const { return *context_.model; }
  const LatticeBase& lattice() const { return *context_.lattice; }
  const Walker& walker() const { return *context_.walker; }
  const MeasureHandler& handler() const { return *context_.handler; }
  const CheckerBoardBase* checkerboard() const { return context_.checkerboard.get(); }

 private:
  // ------------------------------------ Private subroutines
  // -------------------------------------
  void sweep_forth_and_back(std::default_random_engine& rng);

  // ------------------------------------ Member Variables
  // -------------------------------------
  Context context_;  // Owns all the simulation components

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
