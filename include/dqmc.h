#pragma once

/**
 *  This header file defines DQMC::Dqmc class for the organizations
 * of the dqmc program. Top-level designs for the dqmc simulation, e.g.
 * thermalization, measurements and data analysis, are implemented as static
 * member functions of this Dqmc class. Some other useful tools, such as timer
 * and progress bar, are also provided under the namespace
 * DQMC::Dqmc .
 */

#include <chrono>
#include <random>

namespace Model {
class ModelBase;
}
namespace Lattice {
class LatticeBase;
}
namespace Measure {
class MeasureHandler;
}

namespace DQMC {

// forward declaration
class Walker;

using ModelBase = Model::ModelBase;
using LatticeBase = Lattice::LatticeBase;
using MeasureHandler = Measure::MeasureHandler;

// -------------------------------- Pure interface class DQMC::Dqmc
// --------------------------------
class Dqmc {
 public:
  // ---------------------------------------- Useful tools
  // ------------------------------------------

  // set up whether to show the process bar or not
  static void show_progress_bar(bool show_progress_bar);

  // set up the format of the progress bar
  static void progress_bar_format(unsigned int width, char complete,
                                  char incomplete);

  // set up the rate of refreshing the progress bar
  static void set_refresh_rate(unsigned int refresh_rate);

  // return the duration time of the dqmc process, e.g thermalization or
  // measurements
  static const double timer();

  // start the timer
  static void timer_begin();

  // end the timer
  static void timer_end();

  // ------------------------------------ Crucial Dqmc routines
  // -------------------------------------

  // thermalization of the field configurations
  static void thermalize(Walker& walker, ModelBase& model, LatticeBase& lattice,
                         MeasureHandler& meas_handler,
                         std::default_random_engine& rng);

  // Monte Carlo updates and measurments
  static void measure(Walker& walker, ModelBase& model, LatticeBase& lattice,
                      MeasureHandler& meas_handler,
                      std::default_random_engine& rng);

  // analyse the measured data
  static void analyse(MeasureHandler& meas_handler);

 private:
  // ------------------------------------ Private subroutines
  // ------------------------------------- declarations of static members
  static bool m_show_progress_bar;
  static unsigned int m_progress_bar_width;
  static unsigned int m_refresh_rate;
  static char m_progress_bar_complete_char, m_progress_bar_incomplete_char;

  static std::chrono::steady_clock::time_point m_begin_time, m_end_time;

  // sweep and update the field configurations
  // from 0 to beta and back from beta to 0
  // do the measurements if needed
  static void sweep_forth_and_back(Walker& walker, ModelBase& model,
                                   LatticeBase& lattice,
                                   MeasureHandler& meas_handler,
                                   std::default_random_engine& rng);
};
}  // namespace DQMC
