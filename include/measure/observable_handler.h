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

    std::function<ptrBaseObs(const ObsName&, const std::string&)> factory;
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

  ptrScalar m_equaltime_sign{};
  ptrScalar m_dynamic_sign{};

  // Central registry of all supported observables and their properties.
  static const std::map<ObsName, ObservableProperties> m_supported_observables;

 public:
  ObservableHandler() = default;

  static std::vector<std::string> get_all_observable_names();

  // find observable and return std::optional
  template <typename ObsType>
  std::optional<ObsType> find(const ObsName& obs_name);

  // check if certain observable is of scalar type
  bool is_scalar(const ObsName& obs_name) const;

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
template <typename ObsType>
std::optional<ObsType> ObservableHandler::find(const ObsName& obs_name) {
  auto it = this->m_obs_map.find(obs_name);
  if (it != this->m_obs_map.end()) {
    auto ptrObs = std::dynamic_pointer_cast<ObsType>(it->second);
    if (ptrObs) {
      return *ptrObs;
    }
  }
  return std::nullopt;
}
}  // namespace Observable
