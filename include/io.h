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
  template <typename StreamType>
  static void output_ending_info(StreamType& ostream, const Walker& walker);

  // output the mean value and error bar of one specific observable
  template <typename StreamType, typename ObsType>
  static void output_observable(StreamType& ostream,
                                const Observable::Observable<ObsType>& obs);

  // output the bin data of one specific observable
  template <typename StreamType, typename ObsType>
  static void output_observable_in_bins(
      StreamType& ostream, const Observable::Observable<ObsType>& obs);

  // output list of inequivalent momentum points ( k stars )
  template <typename StreamType>
  static void output_k_stars(StreamType& ostream, const LatticeBase& lattice);

  // output imgainary-time grids
  template <typename StreamType>
  static void output_imaginary_time_grids(StreamType& ostream,
                                          const Walker& walker);

  // output the current configuration the bosonic fields,
  // depending on specific model type
  template <typename StreamType>
  static void output_bosonic_fields(StreamType& ostream,
                                    const ModelBase& model);

  // read the configuration of the bosonic fields from input file
  // depending on specific model type
  static void read_bosonic_fields_from_file(const std::string& filename,
                                            ModelBase& model);
};

// -------------------------------------------------------------------------------------------------------
//                                Implementation of template functions
// -------------------------------------------------------------------------------------------------------

void IO::output_init_info(std::ostream& ostream, int world_size,
                          const ModelBase& model, const LatticeBase& lattice,
                          const Walker& walker,
                          const MeasureHandler& meas_handler,
                          const CheckerBoardBasePtr& checkerboard) {
  if (!ostream) {
    std::cerr << "DQMC::IO::output_init_info(): "
              << "the ostream failed to work, please check the input."
              << std::endl;
    exit(1);
  } else {
    // output formats
    auto fmt_param_str = [](const std::string& desc, const std::string& joiner,
                            const std::string& value) {
      return std::format("{:>30s}{:>7s}{:>24s}\n", desc, joiner, value);
    };
    auto fmt_param_int = [](const std::string& desc, const std::string& joiner,
                            int value) {
      return std::format("{:>30s}{:>7s}{:>24d}\n", desc, joiner, value);
    };
    auto fmt_param_double = [](const std::string& desc,
                               const std::string& joiner, double value) {
      return std::format("{:>30s}{:>7s}{:>24.3f}\n", desc, joiner, value);
    };
    std::string joiner = "->";
    auto bool2str = [](bool b) { return b ? "True" : "False"; };

    // -------------------------------------------------------------------------------------------
    //                                 Output model information
    // -------------------------------------------------------------------------------------------
    model.output_model_info(ostream);

    // -------------------------------------------------------------------------------------------
    //                                Output lattice information
    // -------------------------------------------------------------------------------------------
    lattice.output_lattice_info(ostream, meas_handler.Momentum());

    // -------------------------------------------------------------------------------------------
    //                              Output CheckerBoard information
    // -------------------------------------------------------------------------------------------
    ostream << fmt_param_str("Checkerboard breakups", joiner,
                             bool2str((bool)checkerboard))
            << std::endl;

    // -------------------------------------------------------------------------------------------
    //                               Output MonteCarlo Params
    // -------------------------------------------------------------------------------------------
    ostream << "   MonteCarlo Params:\n"
            << fmt_param_double("Inverse temperature", joiner, walker.Beta())
            << fmt_param_int("Imaginary-time length", joiner, walker.TimeSize())
            << fmt_param_double("Imaginary-time interval", joiner,
                                walker.TimeInterval())
            << fmt_param_int("Stabilization pace", joiner,
                             walker.StabilizationPace())
            << std::endl;

    // -------------------------------------------------------------------------------------------
    //                                Output Measuring Params
    // -------------------------------------------------------------------------------------------
    ostream << "   Measuring Params:\n"
            << fmt_param_str("Warm up", joiner,
                             bool2str(meas_handler.isWarmUp()))
            << fmt_param_str("Equal-time measure", joiner,
                             bool2str(meas_handler.isEqualTime()))
            << fmt_param_str("Dynamical measure", joiner,
                             bool2str(meas_handler.isDynamic()))
            << std::endl;

    ostream << fmt_param_int("Sweeps for warmup", joiner,
                             meas_handler.WarmUpSweeps())
            << fmt_param_int("Number of bins", joiner,
                             (meas_handler.BinsNum() * world_size))
            << fmt_param_int("Sweeps per bin", joiner, meas_handler.BinsSize())
            << fmt_param_int("Sweeps between bins", joiner,
                             meas_handler.SweepsBetweenBins())
            << std::endl;
  }
}

template <typename StreamType>
void IO::output_ending_info(StreamType& ostream, const Walker& walker) {
  if (!ostream) {
    throw std::runtime_error(
        "DQMC::IO::output_ending_info(): output stream is not valid.");
  }

  const auto total_duration = Dqmc::timer_as_duration();

  const auto d = std::chrono::duration_cast<std::chrono::days>(total_duration);
  const auto h = std::chrono::duration_cast<std::chrono::hours>(
      total_duration % std::chrono::days(1));
  const auto m = std::chrono::duration_cast<std::chrono::minutes>(
      total_duration % std::chrono::hours(1));

  const std::chrono::duration<double> fractional_seconds =
      total_duration % std::chrono::minutes(1);

  ostream << std::format(
      "\n>> The simulation finished in {}d {}h {}m {:.2f}s.\n", d.count(),
      h.count(), m.count(), fractional_seconds.count());

  ostream << std::format(">> Maximum of the wrapping error: {:.5e}\n",
                         walker.WrapError());
}

template <typename StreamType, typename ObsType>
void IO::output_observable(StreamType& ostream,
                           const Observable::Observable<ObsType>& obs) {
  if (!ostream) {
    std::cerr << "DQMC::IO::output_observable(): "
              << "the ostream failed to work, please check the input."
              << std::endl;
    exit(1);
  } else {
    // standard screen output
    if constexpr (std::is_same_v<StreamType, std::ostream>) {
      // for scalar observables
      if constexpr (std::is_same_v<ObsType, Observable::ScalarType>) {
        auto fmt_scalar_obs = [](const std::string& desc,
                                 const std::string& joiner, double mean,
                                 double error) {
          return std::format("{:>30s}{:>7s}{:>20.12f}  pm  {:.12f}", desc,
                             joiner, mean, error);
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
        std::cerr << "DQMC::IO::output_observable(): "
                  << "undefined observable type." << std::endl;
        exit(1);
      }
    }

    // standard file output
    else if constexpr (std::is_same_v<StreamType, std::ofstream>) {
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
        auto fmt_size_info = [](int size) {
          return std::format("{:>20d}", size);
        };
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
          return std::format("{:>20d}{:>20d}{:>20.10f}{:>20.10f}{:>20.10f}", i,
                             j, mean, error, rel_error);
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
                                      obs.error_bar()(i, j),
                                      relative_error(i, j))
                    << std::endl;
          }
        }
      }

      // other observable types, raising errors
      else {
        std::cerr << "DQMC::IO::output_observable(): "
                  << "undefined observable type." << std::endl;
        exit(1);
      }
    }

    // others stream types, raising errors
    else {
      std::cerr << "DQMC::IO::output_observable(): "
                << "unsupported type of output stream." << std::endl;
      exit(1);
    }
  }
}

template <typename StreamType, typename ObsType>
void IO::output_observable_in_bins(StreamType& ostream,
                                   const Observable::Observable<ObsType>& obs) {
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

template <typename StreamType>
void IO::output_k_stars(StreamType& ostream, const LatticeBase& lattice) {
  if (!ostream) {
    std::cerr << "DQMC::IO::output_k_stars(): "
              << "the ostream failed to work, please check the input."
              << std::endl;
    exit(1);
  } else {
    // output k stars list
    auto fmt_info = [](int value) { return std::format("{:>20d}", value); };
    auto fmt_kstars = [](double value) {
      return std::format("{:>20.10f}", value);
    };
    ostream << fmt_info(lattice.kStarsNum()) << std::endl;
    // loop for inequivalent momentum points
    for (auto i = 0; i < lattice.kStarsNum(); ++i) {
      ostream << fmt_info(i);
      // loop for axises of the reciprocal lattice
      for (auto axis = 0; axis < lattice.SpaceDim(); ++axis) {
        ostream << fmt_kstars(lattice.Index2Momentum(i, axis));
      }
      ostream << std::endl;
    }
  }
}

template <typename StreamType>
void IO::output_imaginary_time_grids(StreamType& ostream,
                                     const Walker& walker) {
  if (!ostream) {
    std::cerr << "DQMC::IO::output_imaginary_time_grids(): "
              << "the ostream failed to work, please check the input."
              << std::endl;
    exit(1);
  } else {
    // output the imaginary-time grids
    auto fmt_tgrids_info = [](int time_size, double beta, double interval) {
      return std::format("{:>20d}{:>20.5f}{:>20.5f}", time_size, beta,
                         interval);
    };
    auto fmt_tgrids = [](int t, double time_val) {
      return std::format("{:>20d}{:>20.10f}", t, time_val);
    };
    ostream << fmt_tgrids_info(walker.TimeSize(), walker.Beta(),
                               walker.TimeInterval())
            << std::endl;
    for (auto t = 0; t < walker.TimeSize(); ++t) {
      ostream << fmt_tgrids(t, (t * walker.TimeInterval())) << std::endl;
    }
  }
}

template <typename StreamType>
void IO::output_bosonic_fields(StreamType& ostream, const ModelBase& model) {
  if (!ostream) {
    std::cerr << "DQMC::IO::output_bosonic_fields(): "
              << "the ostream failed to work, please check the input."
              << std::endl;
    exit(1);
  } else {
    // note that the IO class should be a friend class of any derived model
    // class to get access to the bosonic fields member

    // ---------------------------------  Repulsive Hubbard model
    // ------------------------------------
    if (const auto repulsive_hubbard =
            dynamic_cast<const Model::RepulsiveHubbard*>(&model);
        repulsive_hubbard != nullptr) {
      // output current configuration of auxiliary bosonic fields
      // for repulsive hubbard model, they are ising-like.
      auto fmt_fields_info = [](int time_size, int space_size) {
        return std::format("{:>20d}{:>20d}", time_size, space_size);
      };
      auto fmt_fields = [](int t, int i, double field) {
        return std::format("{:>20d}{:>20d}{:>20.1f}", t, i, field);
      };
      const int time_size = repulsive_hubbard->m_bosonic_field.rows();
      const int space_size = repulsive_hubbard->m_bosonic_field.cols();

      ostream << fmt_fields_info(time_size, space_size) << std::endl;
      for (auto t = 0; t < time_size; ++t) {
        for (auto i = 0; i < space_size; ++i) {
          ostream << fmt_fields(t, i, repulsive_hubbard->m_bosonic_field(t, i))
                  << std::endl;
        }
      }
    }

    // ---------------------------------  Attractive Hubbard model
    // -----------------------------------
    else if (const auto attractive_hubbard =
                 dynamic_cast<const Model::AttractiveHubbard*>(&model);
             attractive_hubbard != nullptr) {
      // output current configuration of auxiliary bosonic fields
      // for attractive hubbard model, they are ising-like.
      auto fmt_fields_info = [](int time_size, int space_size) {
        return std::format("{:>20d}{:>20d}", time_size, space_size);
      };
      auto fmt_fields = [](int t, int i, double field) {
        return std::format("{:>20d}{:>20d}{:>20.1f}", t, i, field);
      };
      const int time_size = attractive_hubbard->m_bosonic_field.rows();
      const int space_size = attractive_hubbard->m_bosonic_field.cols();

      ostream << fmt_fields_info(time_size, space_size) << std::endl;
      for (auto t = 0; t < time_size; ++t) {
        for (auto i = 0; i < space_size; ++i) {
          ostream << fmt_fields(t, i, attractive_hubbard->m_bosonic_field(t, i))
                  << std::endl;
        }
      }
    }

    // other model types, raising errors
    else {
      std::cerr << "DQMC::IO::output_bosonic_fields(): "
                << "undefined model type." << std::endl;
      exit(1);
    }
  }
}

void IO::read_bosonic_fields_from_file(const std::string& filename,
                                       ModelBase& model) {
  std::ifstream infile(filename, std::ios::in);

  // check whether the ifstream works well
  if (!infile.is_open()) {
    std::cerr << "DQMC::IO::read_bosonic_fields_from_file(): "
              << "fail to open file \'" << filename << "\'." << std::endl;
    exit(1);
  }

  // temporary parameters
  std::string line;
  std::vector<std::string> data;

  // note that the IO class should be a friend class of any derived model
  // class to get access to the bosonic fields member

  // ---------------------------------  Repulsive Hubbard model
  // ------------------------------------
  if (auto repulsive_hubbard = dynamic_cast<Model::RepulsiveHubbard*>(&model);
      repulsive_hubbard != nullptr) {
    // consistency check of the model parameters
    // read the first line which containing the model information
    getline(infile, line);
    boost::split(data, line, boost::is_any_of(" "), boost::token_compress_on);
    data.erase(std::remove(std::begin(data), std::end(data), ""),
               std::end(data));

    const int time_size = boost::lexical_cast<int>(data[0]);
    const int space_size = boost::lexical_cast<int>(data[1]);
    if ((time_size != repulsive_hubbard->m_bosonic_field.rows()) ||
        (space_size != repulsive_hubbard->m_bosonic_field.cols())) {
      std::cerr
          << "DQMC::IO::read_bosonic_fields_from_file(): "
          << "inconsistency between model settings and input configs (time or "
             "space size). "
          << std::endl;
      exit(1);
    }

    // read in the configurations of auxiliary fields
    int time_point, space_point;
    while (getline(infile, line)) {
      boost::split(data, line, boost::is_any_of(" "), boost::token_compress_on);
      data.erase(std::remove(std::begin(data), std::end(data), ""),
                 std::end(data));
      time_point = boost::lexical_cast<int>(data[0]);
      space_point = boost::lexical_cast<int>(data[1]);
      repulsive_hubbard->m_bosonic_field(time_point, space_point) =
          boost::lexical_cast<double>(data[2]);
    }
    // close the file stream
    infile.close();
  }

  // ---------------------------------  Attractive Hubbard model
  // -----------------------------------
  else if (auto attractive_hubbard =
               dynamic_cast<Model::AttractiveHubbard*>(&model);
           attractive_hubbard != nullptr) {
    // consistency check of the model parameters
    // read the first line which containing the model information
    getline(infile, line);
    boost::split(data, line, boost::is_any_of(" "), boost::token_compress_on);
    data.erase(std::remove(std::begin(data), std::end(data), ""),
               std::end(data));

    const int time_size = boost::lexical_cast<int>(data[0]);
    const int space_size = boost::lexical_cast<int>(data[1]);
    if ((time_size != attractive_hubbard->m_bosonic_field.rows()) ||
        (space_size != attractive_hubbard->m_bosonic_field.cols())) {
      std::cerr
          << "DQMC::IO::read_bosonic_fields_from_file(): "
          << "inconsistency between model settings and input configs (time or "
             "space size). "
          << std::endl;
      exit(1);
    }

    // read in the configurations of auxiliary fields
    int time_point, space_point;
    while (getline(infile, line)) {
      boost::split(data, line, boost::is_any_of(" "), boost::token_compress_on);
      data.erase(std::remove(std::begin(data), std::end(data), ""),
                 std::end(data));
      time_point = boost::lexical_cast<int>(data[0]);
      space_point = boost::lexical_cast<int>(data[1]);
      attractive_hubbard->m_bosonic_field(time_point, space_point) =
          boost::lexical_cast<double>(data[2]);
    }
    // close the file stream
    infile.close();
  }

  // other model types, raising errors
  else {
    // close the file stream
    infile.close();
    std::cerr << "DQMC::IO::read_bosonic_fields_from_file(): "
              << "undefined model type." << std::endl;
    exit(1);
  }
}
}  // namespace DQMC
