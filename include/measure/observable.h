#pragma once

/**
 * @brief This header file defines the abstract base class
 *        `Observable::ObservableBase` and its template-derived class
 *        `Observable::Observable<ObsType>`. These classes support scalar,
 *        vector, and matrix observable types, providing a framework for
 *        measurement, data collection, and statistical analysis.
 */

#include <Eigen/Core>
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "utils/assert.h"

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

template <typename DataType, typename = void>
struct error_bar_calculator {
  static void calculate(DataType&, const DataType&,
                        const std::vector<DataType>&, int) {
    static_assert(
        sizeof(DataType) == 0,
        "Observable::calculate_error_bar(): Unsupported observable type.");
  }
};

template <>
struct error_bar_calculator<ScalarType> {
  static void calculate(ScalarType& error_bar, const ScalarType& mean_value,
                        const std::vector<ScalarType>& bin_data, int bin_num) {
    for (const auto& data : bin_data) {
      error_bar += std::pow(data, 2);
    }
    error_bar /= bin_num;
    error_bar =
        std::sqrt(error_bar - std::pow(mean_value, 2)) / std::sqrt(bin_num - 1);
  }
};

template <typename EigenType>
struct error_bar_calculator<
    EigenType, std::enable_if_t<
                   std::is_base_of_v<Eigen::DenseBase<EigenType>, EigenType>>> {
  static void calculate(EigenType& error_bar, const EigenType& mean_value,
                        const std::vector<EigenType>& bin_data, int bin_num) {
    for (const auto& data : bin_data) {
      error_bar += data.array().square().matrix();
    }
    error_bar /= bin_num;
    error_bar =
        (error_bar.array() - mean_value.array().square()).sqrt().matrix() /
        std::sqrt(bin_num - 1);
  }
};
}  // namespace detail

/**
 * @brief An abstract base class for all observables. `ObservableBase` provides
 *        common properties and functionalities independent of the specific
 *        observable data type, such as name, description, and binning
 *        information. It manages the counting of samples within a bin.
 */
class ObservableBase {
 protected:
  std::string m_name{};  // name of the observable
  std::string m_desc{};  // description of the observable
  int m_bin_num{0};      // total number of bins
  int m_count{0};        // countings within a bin

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

  int operator++() { return ++this->m_count; }
  int operator+=(int i) { return this->m_count += i; }
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
  using Method = void(Observable<DataType>&, const MeasureHandler&,
                      const Walker&, const ModelBase&, const LatticeBase&);

  DataType m_mean_value{};  // statistical mean value
  DataType m_error_bar{};   // estimated error bar
  DataType m_tmp_value{};   // temporary value during sample collections
  DataType m_zero_elem{};   // zero element to clear temporary values

  std::vector<DataType> m_bin_data{};  // collected data in bins

  std::function<Method> m_method{};  // user-defined measuring method

 public:
  Observable() = default;

  explicit Observable(const std::string& name, const std::string& desc,
                      const std::function<Method>& method)
      : ObservableBase(name, desc), m_method(method) {}

  // -------------------------------------  Interface functions
  // ------------------------------------------

  const DataType& zero_element() const { return this->m_zero_elem; }
  const DataType& mean_value() const { return this->m_mean_value; }
  const DataType& error_bar() const { return this->m_error_bar; }

  const DataType& tmp_value() const { return this->m_tmp_value; }
  DataType& tmp_value() { return this->m_tmp_value; }

  const DataType& bin_data(int bin) const {
    DQMC_ASSERT(bin >= 0 && bin < this->m_bin_num);
    return this->m_bin_data[bin];
  }

  DataType& bin_data(int bin) {
    DQMC_ASSERT(bin >= 0 && bin < this->m_bin_num);
    return this->m_bin_data[bin];
  }

  std::vector<DataType>& bin_data() { return this->m_bin_data; }

  // ---------------------------------  Set up parameters and methods
  // ------------------------------------

  void set_zero_element(const DataType& zero_elem) {
    this->m_zero_elem = zero_elem;
  }

  // -------------------------------------  Other member functions
  // ---------------------------------------

  // perform one step of measurement
  void measure(const MeasureHandler& meas_handler, const Walker& walker,
               const ModelBase& model, const LatticeBase& lattice) {
    this->m_method(*this, meas_handler, walker, model, lattice);
  }

  // allocate memory
  void allocate() {
    this->m_mean_value = this->m_zero_elem;
    this->m_error_bar = this->m_zero_elem;
    this->m_tmp_value = this->m_zero_elem;

    std::vector<DataType>().swap(this->m_bin_data);
    this->m_bin_data.reserve(this->m_bin_num);
    for (int i = 0; i < this->m_bin_num; ++i) {
      this->m_bin_data.emplace_back(this->m_zero_elem);
    }
  }

  // clear statistical data, preparing for a new measurement
  void clear_stats() {
    this->m_mean_value = this->m_zero_elem;
    this->m_error_bar = this->m_zero_elem;
  }

  // clear temporary data
  void clear_temporary() {
    this->m_tmp_value = this->m_zero_elem;
    this->m_count = 0;
  }

  // clear data of bin collections
  void clear_bin_data() {
    for (auto& bin_data : this->m_bin_data) {
      bin_data = this->m_zero_elem;
    }
  }

  // perform data analysis, especially computing the mean and error
  void analyse() {
    this->clear_stats();
    this->calculate_mean_value();
    this->calculate_error_bar();
  }

 private:
  // calculating mean value of the measurement
  void calculate_mean_value() {
    this->m_mean_value = std::accumulate(
        this->m_bin_data.begin(), this->m_bin_data.end(), this->m_zero_elem);
    this->m_mean_value /= this->bin_num();
  }

  // estimate error bar of the measurement
  void calculate_error_bar() {
    detail::error_bar_calculator<DataType>::calculate(
        this->m_error_bar, this->m_mean_value, this->m_bin_data,
        this->bin_num());
  }
};

// some aliases
using Scalar = Observable<ScalarType>;
using Vector = Observable<VectorType>;
using Matrix = Observable<MatrixType>;
}  // namespace Observable
