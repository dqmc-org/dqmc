#pragma once

/**
 *  This header file defines the Observable::ObservableHandler class
 *  to handle with the list of observables during dqmc simulations.
 *  Automatic classification of input observables and the allocation of memory
 * are included.
 */

#include <map>
#include <memory>
#include <optional>

#include "measure/observable.h"

namespace Observable {

using ObsName = std::string;
using ObsNameList = std::vector<std::string>;
using ObsTable = std::vector<std::string>;

enum class ObsTimeType { EqualTime, Dynamic };
enum class ObsDataType { Scalar, Vector, Matrix };

class ObservableHandler {
 protected:
  struct ObservableProperties {
    using ptrBaseObs = std::shared_ptr<ObservableBase>;

    std::string description;
    ObsTimeType time_type;
    ObsDataType data_type;

    std::function<ptrBaseObs(ObsName, const std::string&)> factory;
  };

  using ObsMap = std::map<std::string, std::shared_ptr<ObservableBase>>;

  using ptrBaseObs = std::shared_ptr<ObservableBase>;
  using ptrScalar = std::shared_ptr<Scalar>;
  using ptrVector = std::shared_ptr<Vector>;
  using ptrMatrix = std::shared_ptr<Matrix>;

  using EqtimeScalar = std::vector<std::shared_ptr<Scalar>>;
  using EqtimeVector = std::vector<std::shared_ptr<Vector>>;
  using EqtimeMatrix = std::vector<std::shared_ptr<Matrix>>;
  using DynamicScalar = std::vector<std::shared_ptr<Scalar>>;
  using DynamicVector = std::vector<std::shared_ptr<Vector>>;
  using DynamicMatrix = std::vector<std::shared_ptr<Matrix>>;

  // map of observable objects for quick references
  // only for finding or searching certain observable, and frequent calls should
  // be avoided
  ObsMap m_obs_map{};

  EqtimeScalar m_eqtime_scalar_obs{};
  EqtimeVector m_eqtime_vector_obs{};
  EqtimeMatrix m_eqtime_matrix_obs{};

  DynamicScalar m_dynamic_scalar_obs{};
  DynamicVector m_dynamic_vector_obs{};
  DynamicMatrix m_dynamic_matrix_obs{};

  // Central registry of all supported observables and their properties.
  static const std::map<std::string, ObservableProperties> m_supported_observables;

 public:
  ObservableHandler() = default;

  static std::vector<std::string> get_all_observable_names();

  // find observable and return std::optional
  template <typename DataType>
  std::optional<DataType> find(const std::string& obs_name);

  // initialize the handler
  void initial(const ObsNameList& obs_list);

 private:
  // check if certain observable is of eqtime/dynamic type
  bool is_eqtime(const ObsName& obs_name) const;
  bool is_dynamic(const ObsName& obs_name) const;

  // check the validity of the input list of observables
  bool check_validity(const ObsNameList& obs_list) const;

  // deallocate memory
  void deallocate();
};

// implementation of the template member function
template <typename DataType>
std::optional<DataType> ObservableHandler::find(const std::string& obs_name) {
  if (auto it = this->m_obs_map.find(obs_name); it != this->m_obs_map.end()) {
    if (auto p = std::dynamic_pointer_cast<DataType>(it->second)) {
      return *p;
    }
  }
  return std::nullopt;
}
}  // namespace Observable
