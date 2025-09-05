#pragma once

/**
 * @brief This header file defines the abstract base class
 *        `Observable::ObservableBase` and its template-derived class
 *        `Observable::Observable<ObsType>`. These classes support scalar,
 *        vector, and matrix observable types, providing a framework for
 *        measurement, data collection, and statistical analysis.
 */

#include <Eigen/Core>
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

#include "measure/measure_context.h"
#include "utils/assert.h"
#include "utils/eigen_malloc_guard.h"
#include "utils/temporary_pool.h"

// forward declaration
namespace DQMC {
class Walker;
}
namespace Model {
class ModelBase;
}
namespace Lattice {
class LatticeBase;
}
namespace Measure {
class MeasureHandler;
}

namespace Observable {

using ScalarType = double;
using VectorType = Eigen::VectorXd;
using MatrixType = Eigen::MatrixXd;

namespace detail {
template <typename T>
void calculate_error_bar(T& error_bar, const T& mean_value, const std::vector<T>& bin_data) {
  if constexpr (std::is_floating_point<T>{}) {
    for (const auto& data : bin_data) {
      error_bar += std::pow(data, 2);
    }
    error_bar /= bin_data.size();
    const auto variance = error_bar - std::pow(mean_value, 2);
    if (variance < 0.0 || bin_data.size() <= 1) {
      error_bar = 0.0;
    } else {
      error_bar = std::sqrt(variance) / std::sqrt(bin_data.size() - 1);
    }
  } else if constexpr (std::is_base_of_v<Eigen::MatrixBase<T>, T>) {
    for (const auto& data : bin_data) {
      error_bar += data.array().square().matrix();
    }
    error_bar /= bin_data.size();
    error_bar = (error_bar.array() - mean_value.array().square()).sqrt().matrix() /
                std::sqrt(bin_data.size() - 1);
  }
}

template <typename Derived>
typename Derived::PlainObject zeros_like(const Eigen::MatrixBase<Derived>& expr) {
  EigenMallocGuard<true> alloc_guard;
  using PlainObjectType = typename Derived::PlainObject;
  return PlainObjectType::Zero(expr.rows(), expr.cols());
}

template <std::floating_point Scalar>
Scalar zeros_like(const Scalar&) {
  return static_cast<Scalar>(0);
}
}  // namespace detail

/**
 * @brief An abstract base class for all observables. `ObservableBase` provides
 *        common properties and functionalities independent of the specific
 *        observable data type, such as name, description, and binning
 *        information. It manages the counting of samples within a bin.
 */
class ObservableBase {
 protected:
  std::string m_name;  // name of the observable
  std::string m_desc;  // description of the observable
  int m_bin_num{0};    // total number of bins
  int m_count{0};      // countings within a bin

  ObservableBase() = default;

  explicit ObservableBase(const std::string& name, const std::string& desc)
      : m_name(name), m_desc(desc) {}

 public:
  virtual ~ObservableBase() = default;

  const std::string& name() const { return this->m_name; }
  const std::string& description() const { return this->m_desc; }
  int bin_num() const { return this->m_bin_num; }
  void set_number_of_bins(int bin_num) { this->m_bin_num = bin_num; }

  int counts() const { return this->m_count; }
};

/**
 * @brief A template-derived class from `ObservableBase` that handles the
 *        measurement, storage, and statistical analysis for a specific
 *        observable type. `Observable<ObsType>` manages data collection across
 *        multiple bins, computes mean values, and estimates error bars.
 *
 * @tparam DataType The data type of the observable (e.g., `ScalarType`,
 * `VectorType`, `MatrixType`).
 */
template <typename DataType>
class Observable : public ObservableBase {
 private:
  // useful aliases
  using Walker = DQMC::Walker;
  using ModelBase = Model::ModelBase;
  using LatticeBase = Lattice::LatticeBase;
  using MeasureHandler = Measure::MeasureHandler;
  using Method = void(Observable<DataType>&, const Measure::MeasureContext&);

  DataType m_mean_value{};   // statistical mean value
  DataType m_error_bar{};    // estimated error bar
  DataType m_accumulator{};  // temporary value during sample collections

  std::vector<DataType> m_block_averages{};  // time-series of block averages
  std::vector<DataType> m_final_bins{};      // final binned data for file output

  std::function<Method> m_method{};  // user-defined measuring method

 public:
  Observable() = default;

  explicit Observable(const std::string& name, const std::string& desc,
                      const std::function<Method>& method)
      : ObservableBase(name, desc), m_method(method) {}

  // -------------------------------------  Interface functions
  // ------------------------------------------

  const DataType& mean_value() const { return this->m_mean_value; }
  const DataType& error_bar() const { return this->m_error_bar; }

  const DataType& accumulator() const { return this->m_accumulator; }
  DataType& accumulator() { return this->m_accumulator; }

  void accumulate(const DataType& data, size_t count = 1) {
    m_accumulator += data;
    m_count += count;
  }

  const DataType& bin_data(int bin) const {
    DQMC_ASSERT(bin >= 0 && bin < static_cast<int>(this->m_final_bins.size()));
    return this->m_final_bins[bin];
  }

  DataType& bin_data(int bin) {
    DQMC_ASSERT(bin >= 0 && bin < static_cast<int>(this->m_final_bins.size()));
    return this->m_final_bins[bin];
  }

  std::vector<DataType>& bin_data() { return this->m_final_bins; }

  void start_new_block() { this->clear_temporary(); }

  void finalize_block() {
    EigenMallocGuard<true> alloc_guard;
    this->m_block_averages.push_back(this->m_accumulator);
  }

  const DataType& get_last_block_average() const {
    DQMC_ASSERT(!this->m_block_averages.empty());
    return this->m_block_averages.back();
  }

  void create_final_bins(int optimal_bin_size_blocks) {
    EigenMallocGuard<true> alloc_guard;
    const int num_blocks = static_cast<int>(this->m_block_averages.size());

    if (optimal_bin_size_blocks <= 0 || optimal_bin_size_blocks > num_blocks) {
      // Fallback: treat each block as a bin
      this->m_final_bins = this->m_block_averages;
      this->set_number_of_bins(num_blocks);
    } else {
      // Create bins from blocks using optimal bin size
      const int num_final_bins = num_blocks / optimal_bin_size_blocks;
      this->m_final_bins.clear();
      this->m_final_bins.reserve(num_final_bins);

      for (int i = 0; i < num_final_bins; ++i) {
        DataType bin_sum = detail::zeros_like(m_accumulator);
        for (int j = 0; j < optimal_bin_size_blocks; ++j) {
          bin_sum += this->m_block_averages[i * optimal_bin_size_blocks + j];
        }
        EigenMallocGuard<true> alloc_guard;
        this->m_final_bins.push_back(bin_sum / optimal_bin_size_blocks);
      }
      this->set_number_of_bins(num_final_bins);
    }
  }

  // -------------------------------------  Other member functions
  // ---------------------------------------

  // perform one step of measurement
  void measure(const MeasureHandler& meas_handler, const Walker& walker, const ModelBase& model,
               const LatticeBase& lattice, Utils::TemporaryPool& pool) {
    Measure::MeasureContext ctx(meas_handler, walker, model, lattice, pool);
    this->m_method(*this, ctx);
  }

  // clear statistical data, preparing for a new measurement
  void clear_stats() {
    this->m_mean_value = detail::zeros_like(m_accumulator);
    this->m_error_bar = detail::zeros_like(m_accumulator);
  }

  // clear temporary data
  void clear_temporary() {
    this->m_accumulator = detail::zeros_like(m_accumulator);
    this->m_count = 0;
  }

  // perform data analysis, especially computing the mean and error
  void analyse(int optimal_bin_size_blocks) {
    this->create_final_bins(optimal_bin_size_blocks);
    this->clear_stats();
    this->calculate_mean_value();
    this->calculate_error_bar();
  }

 private:
  // calculating mean value of the measurement
  void calculate_mean_value() {
    this->m_mean_value = std::accumulate(this->m_final_bins.begin(), this->m_final_bins.end(),
                                         detail::zeros_like(m_accumulator));
    this->m_mean_value /= this->m_final_bins.size();
  }

  // estimate error bar of the measurement
  void calculate_error_bar() {
    detail::calculate_error_bar(this->m_error_bar, this->m_mean_value, this->m_final_bins);
  }
};

// some aliases
using Scalar = Observable<ScalarType>;
using Vector = Observable<VectorType>;
using Matrix = Observable<MatrixType>;

template <typename ObsType>
void output_observable_to_console(std::ostream& ostream, const Observable<ObsType>& obs) {
  if (!ostream) {
    throw std::runtime_error(dqmc_format_error("output stream is not valid."));
  }

  // for scalar observables
  if constexpr (std::is_same_v<ObsType, ScalarType>) {
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
    throw std::runtime_error(dqmc_format_error("undefined observable type."));
  }
}

template <typename ObsType>
void output_observable_to_file(std::ofstream& ostream, const Observable<ObsType>& obs) {
  if (!ostream) {
    throw std::runtime_error(dqmc_format_error("output stream is not valid."));
  }

  // for scalar observables
  if constexpr (std::is_same_v<ObsType, ScalarType>) {
    // for specfic scalar observable, output the mean value, error bar and
    // relative error in order.
    ostream << std::format("{:>20.10f}{:>20.10f}{:>20.10f}\n", obs.mean_value(), obs.error_bar(),
                           obs.error_bar() / obs.mean_value());
  }

  // for vector observables
  else if constexpr (std::is_same_v<ObsType, VectorType>) {
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
  else if constexpr (std::is_same_v<ObsType, MatrixType>) {
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
    throw std::runtime_error(dqmc_format_error("undefined observable type."));
  }
}

template <typename ObsType>
void output_observable_in_bins_to_file(std::ofstream& ostream, const Observable<ObsType>& obs) {
  if (!ostream) {
    throw std::runtime_error(
        dqmc_format_error("the ostream failed to work, please check the input."));
  }

  // for scalar observables
  if constexpr (std::is_same_v<ObsType, ScalarType>) {
    // output bin data of scalar observable
    const int number_of_bins = obs.bin_num();
    ostream << std::format("{:>20d}\n", number_of_bins);
    for (auto bin = 0; bin < number_of_bins; ++bin) {
      ostream << std::format("{:>20d}{:>20.10f}\n", bin, obs.bin_data(bin));
    }
  }

  // for vector observables
  else if constexpr (std::is_same_v<ObsType, VectorType>) {
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
  else if constexpr (std::is_same_v<ObsType, MatrixType>) {
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
    throw std::runtime_error(dqmc_format_error("undefined observable type."));
  }
}

}  // namespace Observable
