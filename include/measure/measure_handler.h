#pragma once

/**
 *  This header file defines the measuring handler class Measure::MeasureHandler
 *  to handle with the measuring process during the Monte Carlo updates,
 *  which is derived from another handler class Observable::ObservableHandler.
 */

#include "measure/observable_handler.h"
#include "utils/assert.h"
#include "utils/temporary_pool.h"

// forward declaration
namespace Model {
class ModelBase;
}
namespace Lattice {
class LatticeBase;
}
namespace DQMC {
class Walker;
}

namespace Measure {

using ModelBase = Model::ModelBase;
using LatticeBase = Lattice::LatticeBase;
using Walker = DQMC::Walker;

// -----------------------------------  Handler class Measure::MeasureHandler
// ---------------------------------
class MeasureHandler : public Observable::ObservableHandler {
 private:
  bool m_is_warmup{};     // whether to warm up the system or not
  bool m_is_equaltime{};  // whether to perform equal-time measurements or not
  bool m_is_dynamic{};    // whether to perform dynamic measurements or not

  int m_sweeps_warmup{};  // number of the MC sweeps for the warm-up process

  std::vector<std::string> m_obs_list{};  // list of observables to be measured

  // lattice momentum for the momentum-dependent measurements
  int m_momentum{};
  std::vector<int> m_momentum_list{};

  Utils::TemporaryPool m_pool{};

 public:
  explicit MeasureHandler(int sweeps_warmup, const std::vector<std::string>& observables,
                          int measured_momentum_idx,
                          const std::vector<int>& measured_momentum_list);

  MeasureHandler() = delete;
  MeasureHandler(const MeasureHandler&) = delete;
  MeasureHandler& operator=(const MeasureHandler&) = delete;
  MeasureHandler(MeasureHandler&&) = delete;
  MeasureHandler& operator=(MeasureHandler&&) = delete;

 private:
  // ---------------------------  Set up measuring params and observables
  // ------------------------------

  void set_measure_params(int sweeps_warmup);

  void set_observables(const std::vector<std::string>& obs_list);

  // set up lattice momentum params for momentum-dependent measurements
  // the input momentum list should be provided by Lattice module
  void set_measured_momentum(int momentum_index);
  void set_measured_momentum_list(const std::vector<int>& momentum_index_list);

 public:
  // ------------------------------------  Initializations
  // ---------------------------------------------

  void initial(const LatticeBase& lattice, int time_size);

  // ---------------------------------------  Interfaces
  // -----------------------------------------------

  bool is_warmup() const { return this->m_is_warmup; }
  bool is_equaltime() const { return this->m_is_equaltime; }
  bool is_dynamic() const { return this->m_is_dynamic; }

  int warm_up_sweeps() const { return this->m_sweeps_warmup; }

  int momentum() const { return this->m_momentum; }

  const std::vector<int>& momentum_list() const { return this->m_momentum_list; }

  int momentum_list(const int i) const {
    DQMC_ASSERT(i >= 0 && i < (int)this->m_momentum_list.size());
    return this->m_momentum_list[i];
  }

  const std::vector<std::string>& observables_list() const { return this->m_obs_list; }

  // --------------------------  Subroutines for measuring observables
  // ---------------------------------

  // perform one step of sampling for the measurements
  void equaltime_measure(const Walker& walker, const ModelBase& model, const LatticeBase& lattice);
  void dynamic_measure(const Walker& walker, const ModelBase& model, const LatticeBase& lattice);

  // normalize the observable samples
  void normalize_stats();

  // New block lifecycle methods
  void start_new_block();
  void finalize_block();
  double get_last_block_avg(const std::string& obs_name) const;
  void analyse(int optimal_bin_size_blocks);

  // clear the temporary data
  void clear_temporary();

  // output measuring parameters in a self-documenting format
  void output_measuring_info(std::ostream& ostream) const;
};
}  // namespace Measure
