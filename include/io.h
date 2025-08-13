#pragma once

/**
 *  This header file defines DQMC::IO class
 *  containing the basic IO interfaces for the input/output of the dqmc data
 */

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
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
  static void output_init_info(std::ostream& ostream, int world_size,
                               const ModelBase& model,
                               const LatticeBase& lattice, const Walker& walker,
                               const MeasureHandler& meas_handler,
                               const CheckerBoardBasePtr& checkerboard);

  // output the ending information of the simulation,
  // including time cost and wrapping errors
  static void output_ending_info(std::ostream& ostream, const Walker& walker);

  // output observable to console
  template <typename ObsType>
  static void output_observable_to_console(
      std::ostream& ostream, const Observable::Observable<ObsType>& obs);

  // output observable to file
  template <typename ObsType>
  static void output_observable_to_file(
      std::ofstream& ostream, const Observable::Observable<ObsType>& obs);

  // output the bin data of one specific observable
  template <typename ObsType>
  static void output_observable_in_bins_to_file(
      std::ofstream& ostream, const Observable::Observable<ObsType>& obs);

  // output list of inequivalent momentum points ( k stars )
  static void output_k_stars(std::ostream& ostream, const LatticeBase& lattice);

  // output imgainary-time grids
  static void output_imaginary_time_grids(std::ostream& ostream,
                                          const Walker& walker);

  // output the current configuration the bosonic fields,
  // depending on specific model type
  static void output_bosonic_fields(std::ostream& ostream,
                                    const ModelBase& model);

  // read the configuration of the bosonic fields from input file
  // depending on specific model type
  static void read_bosonic_fields_from_file(const std::string& filename,
                                            ModelBase& model);
};

// -------------------------------------------------------------------------------------------------------
//                                Implementation of template functions
// -------------------------------------------------------------------------------------------------------

template <typename ObsType>
void IO::output_observable_to_console(
    std::ostream& ostream, const Observable::Observable<ObsType>& obs) {
  if (!ostream) {
    throw std::runtime_error(
        "DQMC::IO::output_observable_to_console(): "
        "output stream is not valid.");
  }

  // for scalar observables
  if constexpr (std::is_same_v<ObsType, Observable::ScalarType>) {
    auto fmt_scalar_obs = [](const std::string& desc, const std::string& joiner,
                             double mean, double error) {
      return std::format("{:>30s}{:>7s}{:>20.12f}  pm  {:.12f}", desc, joiner,
                         mean, error);
    };
    const std::string joiner = "->";
    ostream << fmt_scalar_obs(obs.description(), joiner, obs.mean_value(),
                              obs.error_bar())
            << std::endl;
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
    auto fmt_scalar_obs = [](double mean, double error, double rel_error) {
      return std::format("{:>20.10f}{:>20.10f}{:>20.10f}", mean, error,
                         rel_error);
    };
    ostream << fmt_scalar_obs(obs.mean_value(), obs.error_bar(),
                              (obs.error_bar() / obs.mean_value()))
            << std::endl;
  }

  // for vector observables
  else if constexpr (std::is_same_v<ObsType, Observable::VectorType>) {
    // output vector observable
    auto fmt_size_info = [](int size) { return std::format("{:>20d}", size); };
    auto fmt_vector_obs = [](int i, double mean, double error,
                             double rel_error) {
      return std::format("{:>20d}{:>20.10f}{:>20.10f}{:>20.10f}", i, mean,
                         error, rel_error);
    };

    const int size = obs.mean_value().size();
    const auto relative_error =
        (obs.error_bar().array() / obs.mean_value().array()).matrix();
    ostream << fmt_size_info(size) << std::endl;
    for (auto i = 0; i < size; ++i) {
      // output the mean value, error bar and relative error in order.
      ostream << fmt_vector_obs(i, obs.mean_value()(i), obs.error_bar()(i),
                                relative_error(i))
              << std::endl;
    }
  }

  // for matrix observables
  else if constexpr (std::is_same_v<ObsType, Observable::MatrixType>) {
    // output matrix observable
    auto fmt_size_info = [](int row, int col) {
      return std::format("{:>20d}{:>20d}", row, col);
    };
    auto fmt_matrix_obs = [](int i, int j, double mean, double error,
                             double rel_error) {
      return std::format("{:>20d}{:>20d}{:>20.10f}{:>20.10f}{:>20.10f}", i, j,
                         mean, error, rel_error);
    };

    const int row = obs.mean_value().rows();
    const int col = obs.mean_value().cols();
    const auto relative_error =
        (obs.error_bar().array() / obs.mean_value().array()).matrix();
    ostream << fmt_size_info(row, col) << std::endl;
    for (auto i = 0; i < row; ++i) {
      for (auto j = 0; j < col; ++j) {
        // output the mean value, error bar and relative error in order.
        ostream << fmt_matrix_obs(i, j, obs.mean_value()(i, j),
                                  obs.error_bar()(i, j), relative_error(i, j))
                << std::endl;
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
void IO::output_observable_in_bins_to_file(
    std::ofstream& ostream, const Observable::Observable<ObsType>& obs) {
  if (!ostream) {
    std::cerr << "DQMC::IO::output_observable_in_bins(): "
              << "the ostream failed to work, please check the input."
              << std::endl;
    exit(1);
  } else {
    // for scalar observables
    if constexpr (std::is_same_v<ObsType, Observable::ScalarType>) {
      // output bin data of scalar observable
      auto fmt_size_info = [](int bins) {
        return std::format("{:>20d}", bins);
      };
      auto fmt_scalar_obs = [](int bin, double value) {
        return std::format("{:>20d}{:>20.10f}", bin, value);
      };

      const int number_of_bins = obs.bin_num();
      ostream << fmt_size_info(number_of_bins) << std::endl;
      for (auto bin = 0; bin < number_of_bins; ++bin) {
        ostream << fmt_scalar_obs(bin, obs.bin_data(bin)) << std::endl;
      }
    }

    // for vector observables
    else if constexpr (std::is_same_v<ObsType, Observable::VectorType>) {
      // output bin data of vector observable
      auto fmt_size_info = [](int bins, int size) {
        return std::format("{:>20d}{:>20d}", bins, size);
      };
      auto fmt_vector_obs = [](int bin, int i, double value) {
        return std::format("{:>20d}{:>20d}{:>20.10f}", bin, i, value);
      };

      const int number_of_bins = obs.bin_num();
      const int size = obs.mean_value().size();
      ostream << fmt_size_info(number_of_bins, size) << std::endl;
      for (auto bin = 0; bin < number_of_bins; ++bin) {
        for (auto i = 0; i < size; ++i) {
          ostream << fmt_vector_obs(bin, i, obs.bin_data(bin)(i)) << std::endl;
        }
      }
    }

    // for matrix observables
    else if constexpr (std::is_same_v<ObsType, Observable::MatrixType>) {
      // output bin data of matrix observable
      auto fmt_size_info = [](int bins, int row, int col) {
        return std::format("{:>20d}{:>20d}{:>20d}", bins, row, col);
      };
      auto fmt_matrix_obs = [](int bin, int i, int j, double value) {
        return std::format("{:>20d}{:>20d}{:>20d}{:>20.10f}", bin, i, j, value);
      };

      const int number_of_bins = obs.bin_num();
      const int row = obs.mean_value().rows();
      const int col = obs.mean_value().cols();
      ostream << fmt_size_info(number_of_bins, row, col) << std::endl;
      for (auto bin = 0; bin < number_of_bins; ++bin) {
        for (auto i = 0; i < row; ++i) {
          for (auto j = 0; j < col; ++j) {
            ostream << fmt_matrix_obs(bin, i, j, obs.bin_data(bin)(i, j))
                    << std::endl;
          }
        }
      }
    }

    // other observable types, raising errors
    else {
      std::cerr << "DQMC::IO::output_observable_in_bins(): "
                << "undefined observable type." << std::endl;
      exit(1);
    }
  }
}
}  // namespace DQMC
