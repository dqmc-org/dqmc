#pragma once

/**
 *  This header file defines the Observable::ObservableHandler class
 *  to handle with the list of observables during dqmc simulations.
 *  Automatic classification of input observables and the allocation of memory
 * are included.
 */

#include <memory>
#include <unordered_map>

#include "measure/observable.h"

namespace Observable {

enum class ObsTimeType { EqualTime, Dynamic };
enum class ObsDataType { Scalar, Vector, Matrix };

class ObservableHandler {
 protected:
  struct ObservableProperties {
    using ptrBaseObs = std::unique_ptr<ObservableBase>;

    std::string description;
    ObsTimeType time_type;
    ObsDataType data_type;

    std::function<ptrBaseObs(std::string, const std::string&)> factory;
  };

  using ObsMap = std::unordered_map<std::string, std::unique_ptr<ObservableBase>>;

  using EqtimeScalar = std::vector<Scalar*>;
  using EqtimeVector = std::vector<Vector*>;
  using EqtimeMatrix = std::vector<Matrix*>;
  using DynamicScalar = std::vector<Scalar*>;
  using DynamicVector = std::vector<Vector*>;
  using DynamicMatrix = std::vector<Matrix*>;

  ObsMap m_obs_map{};

  EqtimeScalar m_eqtime_scalar_obs{};
  EqtimeVector m_eqtime_vector_obs{};
  EqtimeMatrix m_eqtime_matrix_obs{};

  DynamicScalar m_dynamic_scalar_obs{};
  DynamicVector m_dynamic_vector_obs{};
  DynamicMatrix m_dynamic_matrix_obs{};

  // Central registry of all supported observables and their properties.
  static const std::unordered_map<std::string, ObservableProperties> m_supported_observables;

 public:
  ObservableHandler() = default;

  static std::vector<std::string> get_all_observable_names();

  // find observable and return raw pointer (or nullptr if not found)
  template <typename DataType>
  DataType* find(const std::string& obs_name) const;

  // initialize the handler
  void initial(std::vector<std::string> obs_list);

 private:
  // check if certain observable is of eqtime/dynamic type
  bool is_eqtime(const std::string& obs_name) const;
  bool is_dynamic(const std::string& obs_name) const;

  // check the validity of the input list of observables
  bool check_validity(const std::vector<std::string>& obs_list) const;
};

template <typename DataType>
DataType* ObservableHandler::find(const std::string& obs_name) const {
  if (auto it = this->m_obs_map.find(obs_name); it != this->m_obs_map.end()) {
    return dynamic_cast<DataType*>(it->second.get());
  }
  return nullptr;
}
}  // namespace Observable
