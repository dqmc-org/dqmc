#pragma once

/**
 *  This header file defines the measuring handler class Measure::MeasureHandler
 *  to handle with the measuring process during the Monte Carlo updates,
 *  which is derived from another handler class Observable::ObservableHandler.
 */

#include "measure/observable_handler.h"

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

using ObsList = std::vector<std::string>;
using ModelBase = Model::ModelBase;
using LatticeBase = Lattice::LatticeBase;
using Walker = DQMC::Walker;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using MomentumIndex = int;
using MomentumIndexList = std::vector<int>;

// -----------------------------------  Handler class Measure::MeasureHandler
// ---------------------------------
class MeasureHandler : public Observable::ObservableHandler {
 private:
  bool m_is_warmup{};     // whether to warm up the system or not
  bool m_is_equaltime{};  // whether to perform equal-time measurements or not
  bool m_is_dynamic{};    // whether to perform dynamic measurements or not

  int m_sweeps_warmup{};  // number of the MC sweeps for the warm-up process
  int m_bin_num{};        // number of measuring bins
  int m_bin_size{};       // number of samples in one measuring bin
  int m_sweeps_between_bins{};  // number of the MC sweeps between two adjoining
                                // bins

  ObsList m_obs_list{};  // list of observables to be measured

  // lattice momentum for the momentum-dependent measurements
  MomentumIndex m_momentum{};
  MomentumIndexList m_momentum_list{};

 public:
  MeasureHandler() = default;

  // ---------------------------  Set up measuring params and observables
  // ------------------------------

  void set_measure_params(int sweeps_warmup, int bin_num, int bin_size,
                          int sweeps_between_bins);

  void set_observables(ObsList obs_list);

  // set up lattice momentum params for momentum-dependent measurements
  // the input momentum list should be provided by Lattice module
  void set_measured_momentum(const MomentumIndex& momentum_index);
  void set_measured_momentum_list(const MomentumIndexList& momentum_index_list);

  // ------------------------------------  Initializations
  // ---------------------------------------------

  void initial(const LatticeBase& lattice, const Walker& walker);

  // ---------------------------------------  Interfaces
  // -----------------------------------------------

  bool isWarmUp() const { return this->m_is_warmup; }
  bool isEqualTime() const { return this->m_is_equaltime; }
  bool isDynamic() const { return this->m_is_dynamic; }

  int WarmUpSweeps() const { return this->m_sweeps_warmup; }

  int SweepsBetweenBins() const {
  return this->m_sweeps_between_bins;
}

  int BinsNum() const { return this->m_bin_num; }
  int BinsSize() const { return this->m_bin_size; }

  const MomentumIndex& Momentum() const {
    return this->m_momentum;
  }

  const MomentumIndexList& MomentumList() const {
    return this->m_momentum_list;
  }

  const MomentumIndex& MomentumList(const int i) const {
    // assert(i >= 0 && i < (int)this->m_momentum_list.size());
  return this->m_momentum_list[i];
  }


  // the following interfaces have been implemented
  // in the base class Observable::ObservableHandler.
  // and can be directly called in the MeasureHandler class.

  // // check if certain observable exists
  // bool find(const ObsName& obs_name);

  // // return certain type of the observable class
  // template<typename ObsType> const ObsType find(const ObsName& obs_name);

  // --------------------------  Subroutines for measuring observables
  // ---------------------------------

  // perform one step of sampling for the measurements
  void equaltime_measure(const Walker& walker, const ModelBase& model,
                         const LatticeBase& lattice);
  void dynamic_measure(const Walker& walker, const ModelBase& model,
                       const LatticeBase& lattice);

  // normalize the observable samples
  void normalize_stats();

  // bin collections of the observable samples
  void write_stats_to_bins(int bin);

  // analyse the statistics by calculating means and errors
  void analyse_stats();

  // clear the temporary data
  void clear_temporary();

  // output measuring parameters in a self-documenting format
  void output_measuring_info(std::ostream& ostream, int world_size) const;
};
}  // namespace Measure
