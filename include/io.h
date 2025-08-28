#pragma once

/**
 *  This header file defines DQMC::IO class
 *  containing the basic IO interfaces for the input/output of the dqmc data
 */

#include <chrono>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>

#include "checkerboard/checkerboard_base.h"
#include "dqmc.h"
#include "lattice/cubic.h"
#include "lattice/honeycomb.h"
#include "lattice/lattice_base.h"
#include "lattice/square.h"
#include "measure/measure_handler.h"
#include "measure/observable.h"
#include "model/attractive_hubbard.h"
#include "model/model_base.h"
#include "model/repulsive_hubbard.h"
#include "walker.h"

namespace DQMC {

using ModelBase = Model::ModelBase;
using LatticeBase = Lattice::LatticeBase;
using MeasureHandler = Measure::MeasureHandler;
using CheckerBoardBasePtr = std::unique_ptr<CheckerBoard::CheckerBoardBase>;

// ---------------------------- IO Interface class DQMC::IO
// ------------------------------
class IO {
 public:
  // output the information of dqmc initialization,
  // including initialization status and simulation parameters.
  // the behavior of this function depends on specific model and lattice types.
  static void output_init_info(std::ostream& ostream, const Dqmc& simulation);

  // output the ending information of the simulation,
  // including time cost and wrapping errors
  static void output_ending_info(std::ostream& ostream, const Walker& walker,
                                 std::chrono::milliseconds duration);

  // output observable to console
  template <typename ObsType>
  static void output_observable_to_console(std::ostream& ostream,
                                           const Observable::Observable<ObsType>& obs);

  // output observable to file
  template <typename ObsType>
  static void output_observable_to_file(std::ofstream& ostream,
                                        const Observable::Observable<ObsType>& obs);

  // output the bin data of one specific observable
  template <typename ObsType>
  static void output_observable_in_bins_to_file(std::ofstream& ostream,
                                                const Observable::Observable<ObsType>& obs);
};

// -------------------------------------------------------------------------------------------------------
//                                Implementation of template functions
// -------------------------------------------------------------------------------------------------------

template <typename ObsType>
void IO::output_observable_to_console(std::ostream& ostream,
                                      const Observable::Observable<ObsType>& obs) {
  if (!ostream) {
    throw std::runtime_error(
        "DQMC::IO::output_observable_to_console(): "
        "output stream is not valid.");
  }

  // for scalar observables
  if constexpr (std::is_same_v<ObsType, Observable::ScalarType>) {
    ostream << std::format("{:>30s}{:>7s}{:>20.12f}  pm  {:.12f}\n", obs.description(), "->",
                           obs.mean_value(), obs.error_bar());
  }

  // // todo: currently not used
  // // for vector observables
  // else if constexpr ( std::is_same_v<ObsType, Observable::VectorType> ) {

  // }

  // // for matrix observables
  // else if constexpr ( std::is_same_v<ObsType, Observable::MatrixType> ) {

  // }

  // other observable type, raising errors
  else {
    throw std::runtime_error(
        "DQMC::IO::output_observable_to_console(): "
        "undefined observable type.");
  }
}

template <typename ObsType>
void IO::output_observable_to_file(std::ofstream& ostream,
                                   const Observable::Observable<ObsType>& obs) {
  if (!ostream) {
    throw std::runtime_error(
        "DQMC::IO::output_observable_to_file(): "
        "output stream is not valid.");
  }

  // for scalar observables
  if constexpr (std::is_same_v<ObsType, Observable::ScalarType>) {
    // for specfic scalar observable, output the mean value, error bar and
    // relative error in order.
    ostream << std::format("{:>20.10f}{:>20.10f}{:>20.10f}\n", obs.mean_value(), obs.error_bar(),
                           obs.error_bar() / obs.mean_value());
  }

  // for vector observables
  else if constexpr (std::is_same_v<ObsType, Observable::VectorType>) {
    // output vector observable
    const int size = obs.mean_value().size();
    const auto relative_error = (obs.error_bar().array() / obs.mean_value().array()).matrix();
    ostream << std::format("{:>20d}", size) << std::endl;
    for (auto i = 0; i < size; ++i) {
      // output the mean value, error bar and relative error in order.
      ostream << std::format("{:>20d}{:>20.10f}{:>20.10f}{:>20.10f}\n", i, obs.mean_value()(i),
                             obs.error_bar()(i), relative_error(i));
    }
  }

  // for matrix observables
  else if constexpr (std::is_same_v<ObsType, Observable::MatrixType>) {
    // output matrix observable
    const int row = obs.mean_value().rows();
    const int col = obs.mean_value().cols();
    const auto relative_error = (obs.error_bar().array() / obs.mean_value().array()).matrix();
    ostream << std::format("{:>20d}{:>20d}", row, col) << std::endl;
    for (auto i = 0; i < row; ++i) {
      for (auto j = 0; j < col; ++j) {
        // output the mean value, error bar and relative error in order.
        ostream << std::format("{:>20d}{:>20d}{:>20.10f}{:>20.10f}{:>20.10f}\n", i, j,
                               obs.mean_value()(i, j), obs.error_bar()(i, j), relative_error(i, j));
      }
    }
  }

  // other observable types, raising errors
  else {
    throw std::runtime_error(
        "DQMC::IO::output_observable_to_file(): "
        "undefined observable type.");
  }
}

template <typename ObsType>
void IO::output_observable_in_bins_to_file(std::ofstream& ostream,
                                           const Observable::Observable<ObsType>& obs) {
  if (!ostream) {
    std::cerr << "DQMC::IO::output_observable_in_bins(): "
              << "the ostream failed to work, please check the input." << std::endl;
    exit(1);
  } else {
    // for scalar observables
    if constexpr (std::is_same_v<ObsType, Observable::ScalarType>) {
      // output bin data of scalar observable
      const int number_of_bins = obs.bin_num();
      ostream << std::format("{:>20d}\n", number_of_bins);
      for (auto bin = 0; bin < number_of_bins; ++bin) {
        ostream << std::format("{:>20d}{:>20.10f}\n", bin, obs.bin_data(bin));
      }
    }

    // for vector observables
    else if constexpr (std::is_same_v<ObsType, Observable::VectorType>) {
      // output bin data of vector observable
      const int number_of_bins = obs.bin_num();
      const int size = obs.mean_value().size();
      ostream << std::format("{:>20d}{:>20d}\n", number_of_bins, size);
      for (auto bin = 0; bin < number_of_bins; ++bin) {
        for (auto i = 0; i < size; ++i) {
          ostream << std::format("{:>20d}{:>20d}{:>20.10f}\n", bin, i, obs.bin_data(bin)(i));
        }
      }
    }

    // for matrix observables
    else if constexpr (std::is_same_v<ObsType, Observable::MatrixType>) {
      // output bin data of matrix observable
      const int number_of_bins = obs.bin_num();
      const int row = obs.mean_value().rows();
      const int col = obs.mean_value().cols();
      ostream << std::format("{:>20d}{:>20d}{:>20d}\n", number_of_bins, row, col);
      for (auto bin = 0; bin < number_of_bins; ++bin) {
        for (auto i = 0; i < row; ++i) {
          for (auto j = 0; j < col; ++j) {
            ostream << std::format("{:>20d}{:>20d}{:>20d}{:>20.10f}\n", bin, i, j,
                                   obs.bin_data(bin)(i, j));
          }
        }
      }
    }

    // other observable types, raising errors
    else {
      std::cerr << "DQMC::IO::output_observable_in_bins(): " << "undefined observable type."
                << std::endl;
      exit(1);
    }
  }
}
}  // namespace DQMC
