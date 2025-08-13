#pragma once

/**
 *  This header file defines the Observable::ObservableHandler class
 *  to handle with the list of observables during dqmc simulations.
 *  Automatic classification of input observables and the allocation of memory
 * are included.
 */

#include <map>
#include <memory>

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
  using ptrScalarObs = std::shared_ptr<ScalarObs>;
  using ptrVectorObs = std::shared_ptr<VectorObs>;
  using ptrMatrixObs = std::shared_ptr<MatrixObs>;

  using EqtimeScalarObs = std::vector<std::shared_ptr<ScalarObs>>;
  using EqtimeVectorObs = std::vector<std::shared_ptr<VectorObs>>;
  using EqtimeMatrixObs = std::vector<std::shared_ptr<MatrixObs>>;
  using DynamicScalarObs = std::vector<std::shared_ptr<ScalarObs>>;
  using DynamicVectorObs = std::vector<std::shared_ptr<VectorObs>>;
  using DynamicMatrixObs = std::vector<std::shared_ptr<MatrixObs>>;

  // map of observable objects for quick references
  // only for finding or searching certain observable, and frequent calls should
  // be avoided
  ObsMap m_obs_map{};

  EqtimeScalarObs m_eqtime_scalar_obs{};
  EqtimeVectorObs m_eqtime_vector_obs{};
  EqtimeMatrixObs m_eqtime_matrix_obs{};

  DynamicScalarObs m_dynamic_scalar_obs{};
  DynamicVectorObs m_dynamic_vector_obs{};
  DynamicMatrixObs m_dynamic_matrix_obs{};

  ptrScalarObs m_equaltime_sign{};
  ptrScalarObs m_dynamic_sign{};

  // Central registry of all supported observables and their properties.
  static const std::map<ObsName, ObservableProperties> m_supported_observables;

 public:
  ObservableHandler() = default;

  static std::vector<std::string> get_all_observable_names();

  // check if certain observable exists
  bool find(const ObsName& obs_name);

  // return certain type of the observable class
  template <typename ObsType>
  const ObsType find(const ObsName& obs_name);

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
const ObsType ObservableHandler::find(const ObsName& obs_name) {
  if (this->find(obs_name)) {
    auto ptrObs = std::dynamic_pointer_cast<ObsType>(this->m_obs_map[obs_name]);
    if (ptrObs) {
      return *ptrObs;
    }
  }
  return ObsType();
}
}  // namespace Observable
