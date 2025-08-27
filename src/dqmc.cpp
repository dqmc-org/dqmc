#include "dqmc.h"

#include "lattice/lattice_base.h"
#include "measure/measure_handler.h"
#include "model/model_base.h"
#include "utils/progressbar.hpp"
#include "walker.h"

namespace DQMC {

Dqmc::Dqmc(Context&& context) : context_(std::move(context)) {}

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

// timer functions
void Dqmc::timer_begin() { m_begin_time = std::chrono::steady_clock::now(); }

void Dqmc::timer_end() { m_end_time = std::chrono::steady_clock::now(); }

double Dqmc::timer() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(m_end_time - m_begin_time).count();
}

std::chrono::milliseconds Dqmc::timer_as_duration() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(m_end_time - m_begin_time);
}

// -----------------------------------  Crucial static member functions
// --------------------------------------
// ---------------------------------  For organizing the dqmc simulations
// ------------------------------------

void Dqmc::sweep_forth_and_back(std::default_random_engine& rng) {
  // sweep forth from 0 to beta
  if (context_.handler->isDynamic()) {
    context_.walker->sweep_for_dynamic_greens(*context_.model);
    context_.handler->dynamic_measure(*context_.walker, *context_.model, *context_.lattice);
  } else {
    context_.walker->sweep_from_0_to_beta(*context_.model, rng);
    if (context_.handler->isEqualTime()) {
      context_.handler->equaltime_measure(*context_.walker, *context_.model, *context_.lattice);
    }
  }

  // sweep back from beta to 0
  context_.walker->sweep_from_beta_to_0(*context_.model, rng);
  if (context_.handler->isEqualTime()) {
    context_.handler->equaltime_measure(*context_.walker, *context_.model, *context_.lattice);
  }
}

void Dqmc::thermalize(std::default_random_engine& rng) {
  if (context_.handler->isWarmUp()) {
    // create progress bar
    ProgressBar progressbar(context_.handler->WarmUpSweeps() / 2, m_progress_bar_width,
                            m_progress_bar_complete_char, m_progress_bar_incomplete_char);

    // warm-up sweeps
    for (auto sweep = 1; sweep <= context_.handler->WarmUpSweeps() / 2; ++sweep) {
      // sweep forth and back without measuring
      context_.walker->sweep_from_0_to_beta(*context_.model, rng);
      context_.walker->sweep_from_beta_to_0(*context_.model, rng);

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

void Dqmc::measure(std::default_random_engine& rng) {
  if (context_.handler->isEqualTime() || context_.handler->isDynamic()) {
    // create progress bar
    ProgressBar progressbar(context_.handler->BinsNum() * context_.handler->BinsSize() / 2,
                            m_progress_bar_width, m_progress_bar_complete_char,
                            m_progress_bar_incomplete_char);

    // measuring sweeps
    for (auto bin = 0; bin < context_.handler->BinsNum(); ++bin) {
      for (auto sweep = 1; sweep <= context_.handler->BinsSize() / 2; ++sweep) {
        // update and measure
        sweep_forth_and_back(rng);

        // record the tick
        ++progressbar;
        if (m_show_progress_bar && (sweep % m_refresh_rate == 1)) {
          std::cout << " Measuring  ";
          progressbar.display();
        }
      }

      // store the collected data in the MeasureHandler
      context_.handler->normalize_stats();
      context_.handler->write_stats_to_bins(bin);
      context_.handler->clear_temporary();

      // avoid correlations between adjoining bins
      for (auto sweep = 0; sweep < context_.handler->SweepsBetweenBins() / 2; ++sweep) {
        context_.walker->sweep_from_0_to_beta(*context_.model, rng);
        context_.walker->sweep_from_beta_to_0(*context_.model, rng);
      }
    }

    // progress bar finish
    if (m_show_progress_bar) {
      std::cout << " Measuring  ";
      progressbar.done();
    }
  }
}

void Dqmc::analyse() {
  // analyse the collected data after the measuring process
  context_.handler->analyse_stats();
}

}  // namespace DQMC
