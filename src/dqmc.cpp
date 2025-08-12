#include "dqmc.h"

#include "lattice/lattice_base.h"
#include "measure/measure_handler.h"
#include "model/model_base.h"
#include "utils/progressbar.hpp"
#include "walker.h"

namespace DQMC {

// definitions of the static members
bool Dqmc::m_show_progress_bar{true};
unsigned int Dqmc::m_progress_bar_width{70};
unsigned int Dqmc::m_refresh_rate{10};
char Dqmc::m_progress_bar_complete_char{'='},
    Dqmc::m_progress_bar_incomplete_char{' '};
std::chrono::steady_clock::time_point Dqmc::m_begin_time{}, Dqmc::m_end_time{};

// set up whether to show the process bar or not
void Dqmc::show_progress_bar(bool show_progress_bar) {
  Dqmc::m_show_progress_bar = show_progress_bar;
}

// set up the format of the progress bar
void Dqmc::progress_bar_format(unsigned int width, char complete,
                               char incomplete) {
  Dqmc::m_progress_bar_width = width;
  Dqmc::m_progress_bar_complete_char = complete;
  Dqmc::m_progress_bar_incomplete_char = incomplete;
}

// set up the rate of refreshing the progress bar
void Dqmc::set_refresh_rate(unsigned int refresh_rate) {
  Dqmc::m_refresh_rate = refresh_rate;
}

// timer functions
void Dqmc::timer_begin() {
  Dqmc::m_begin_time = std::chrono::steady_clock::now();
}

void Dqmc::timer_end() { Dqmc::m_end_time = std::chrono::steady_clock::now(); }

double Dqmc::timer() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             Dqmc::m_end_time - Dqmc::m_begin_time)
      .count();
}

std::chrono::milliseconds Dqmc::timer_as_duration() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
      Dqmc::m_end_time - Dqmc::m_begin_time);
}

// -----------------------------------  Crucial static member functions
// --------------------------------------
// ---------------------------------  For organizing the dqmc simulations
// ------------------------------------

void Dqmc::sweep_forth_and_back(Walker& walker, ModelBase& model,
                                LatticeBase& lattice,
                                MeasureHandler& meas_handler,
                                std::default_random_engine& rng) {
  // sweep forth from 0 to beta
  if (meas_handler.isDynamic()) {
    walker.sweep_for_dynamic_greens(model);
    meas_handler.dynamic_measure(walker, model, lattice);
  } else {
    walker.sweep_from_0_to_beta(model, rng);
    if (meas_handler.isEqualTime()) {
      meas_handler.equaltime_measure(walker, model, lattice);
    }
  }

  // sweep back from beta to 0
  walker.sweep_from_beta_to_0(model, rng);
  if (meas_handler.isEqualTime()) {
    meas_handler.equaltime_measure(walker, model, lattice);
  }
}

void Dqmc::thermalize(Walker& walker, ModelBase& model, LatticeBase& lattice,
                      MeasureHandler& meas_handler,
                      std::default_random_engine& rng) {
  if (meas_handler.isWarmUp()) {
    // create progress bar
    ProgressBar progressbar(
        meas_handler.WarmUpSweeps() / 2,      // total loops
        Dqmc::m_progress_bar_width,           // bar width
        Dqmc::m_progress_bar_complete_char,   // complete character
        Dqmc::m_progress_bar_incomplete_char  // incomplete character
    );

    // warm-up sweeps
    for (auto sweep = 1; sweep <= meas_handler.WarmUpSweeps() / 2; ++sweep) {
      // sweep forth and back without measuring
      walker.sweep_from_0_to_beta(model, rng);
      walker.sweep_from_beta_to_0(model, rng);

      // record the tick
      ++progressbar;
      if (Dqmc::m_show_progress_bar && (sweep % Dqmc::m_refresh_rate == 1)) {
        std::cout << " Warming up ";
        progressbar.display();
      }
    }

    // progress bar finish
    if (Dqmc::m_show_progress_bar) {
      std::cout << " Warming up ";
      progressbar.done();
    }
  }
}

void Dqmc::measure(Walker& walker, ModelBase& model, LatticeBase& lattice,
                   MeasureHandler& meas_handler,
                   std::default_random_engine& rng) {
  if (meas_handler.isEqualTime() || meas_handler.isDynamic()) {
    // create progress bar
    ProgressBar progressbar(
        meas_handler.BinsNum() * meas_handler.BinsSize() / 2,
        Dqmc::m_progress_bar_width, Dqmc::m_progress_bar_complete_char,
        Dqmc::m_progress_bar_incomplete_char);

    // measuring sweeps
    for (auto bin = 0; bin < meas_handler.BinsNum(); ++bin) {
      for (auto sweep = 1; sweep <= meas_handler.BinsSize() / 2; ++sweep) {
        // update and measure
        Dqmc::sweep_forth_and_back(walker, model, lattice, meas_handler, rng);

        // record the tick
        ++progressbar;
        if (Dqmc::m_show_progress_bar && (sweep % Dqmc::m_refresh_rate == 1)) {
          std::cout << " Measuring  ";
          progressbar.display();
        }
      }

      // store the collected data in the MeasureHandler
      meas_handler.normalize_stats();
      meas_handler.write_stats_to_bins(bin);
      meas_handler.clear_temporary();

      // avoid correlations between adjoining bins
      for (auto sweep = 0; sweep < meas_handler.SweepsBetweenBins() / 2;
           ++sweep) {
        walker.sweep_from_0_to_beta(model, rng);
        walker.sweep_from_beta_to_0(model, rng);
      }
    }

    // progress bar finish
    if (Dqmc::m_show_progress_bar) {
      std::cout << " Measuring  ";
      progressbar.done();
    }
  }
}

void Dqmc::analyse(MeasureHandler& meas_handler) {
  // analyse the collected data after the measuring process
  meas_handler.analyse_stats();
}

}  // namespace DQMC
