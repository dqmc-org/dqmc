#include "measure/observable_handler.h"

#include <format>
#include <iostream>

#include "measure/measure_methods.h"

namespace Observable {

namespace {
template <typename ObsType, typename MethodFunc>
auto make_observable(MethodFunc method) {
  return [method](const ObsName& name,
                  const std::string& desc) -> std::shared_ptr<ObservableBase> {
    return std::make_shared<ObsType>(name, desc, method);
  };
}
}  // namespace

// clang-format off
const std::map<ObsName, ObservableHandler::ObservableProperties> ObservableHandler::m_supported_observables = {
    // --- Equal-Time Observables ---
    {"filling_number",                 {"Filling number",              ObsTimeType::EqualTime, ObsDataType::Scalar, make_observable<ScalarObs>(Measure::Methods::measure_filling_number)}},
    {"double_occupancy",               {"Double occupation",           ObsTimeType::EqualTime, ObsDataType::Scalar, make_observable<ScalarObs>(Measure::Methods::measure_double_occupancy)}},
    {"kinetic_energy",                 {"Kinetic energy",              ObsTimeType::EqualTime, ObsDataType::Scalar, make_observable<ScalarObs>(Measure::Methods::measure_kinetic_energy)}},
    {"momentum_distribution",          {"Momentum distribution",       ObsTimeType::EqualTime, ObsDataType::Scalar, make_observable<ScalarObs>(Measure::Methods::measure_momentum_distribution)}},
    {"local_spin_corr",                {"Local spin correlation",      ObsTimeType::EqualTime, ObsDataType::Scalar, make_observable<ScalarObs>(Measure::Methods::measure_local_spin_corr)}},
    {"spin_density_structure_factor",  {"SDW order parameter",         ObsTimeType::EqualTime, ObsDataType::Scalar, make_observable<ScalarObs>(Measure::Methods::measure_spin_density_structure_factor)}},
    {"charge_density_structure_factor",{"CDW order parameter",         ObsTimeType::EqualTime, ObsDataType::Scalar, make_observable<ScalarObs>(Measure::Methods::measure_charge_density_structure_factor)}},
    {"s_wave_pairing_corr",            {"S-wave pairing correlation",  ObsTimeType::EqualTime, ObsDataType::Scalar, make_observable<ScalarObs>(Measure::Methods::measure_s_wave_pairing_corr)}},

    // --- Dynamic Observables ---
    {"greens_functions",               {"Green's functions",           ObsTimeType::Dynamic,   ObsDataType::Matrix, make_observable<MatrixObs>(Measure::Methods::measure_greens_functions)}},
    {"density_of_states",              {"Density of states",           ObsTimeType::Dynamic,   ObsDataType::Vector, make_observable<VectorObs>(Measure::Methods::measure_density_of_states)}},
    {"superfluid_stiffness",           {"Superfluid stiffness",        ObsTimeType::Dynamic,   ObsDataType::Scalar, make_observable<ScalarObs>(Measure::Methods::measure_superfluid_stiffness)}},
    {"dynamic_spin_susceptibility",    {"Dynamic Spin Susceptibility", ObsTimeType::Dynamic,   ObsDataType::Vector, make_observable<VectorObs>(Measure::Methods::measure_dynamic_spin_susceptibility)}},
};
// clang-format on

std::vector<std::string> ObservableHandler::get_all_observable_names() {
  std::vector<std::string> names;
  names.reserve(m_supported_observables.size());
  for (const auto& pair : m_supported_observables) {
    names.push_back(pair.first);
  }
  return names;
}

bool ObservableHandler::is_eqtime(const ObsName& obs_name) const {
  auto it = m_supported_observables.find(obs_name);
  return it != m_supported_observables.end() &&
         it->second.time_type == ObsTimeType::EqualTime;
}

bool ObservableHandler::is_dynamic(const ObsName& obs_name) const {
  auto it = m_supported_observables.find(obs_name);
  return it != m_supported_observables.end() &&
         it->second.time_type == ObsTimeType::Dynamic;
}

bool ObservableHandler::check_validity(const ObsNameList& obs_list) const {
  for (const auto& obs : obs_list) {
    if (m_supported_observables.find(obs) == m_supported_observables.end()) {
      std::string error_message =
          std::format("Error: observable '{}' is not supported.", obs);
      throw std::runtime_error(error_message);
    }
  }
  return true;
}

void ObservableHandler::deallocate() {
  this->m_obs_map.clear();

  this->m_eqtime_scalar_obs.clear();
  this->m_eqtime_vector_obs.clear();
  this->m_eqtime_matrix_obs.clear();
  this->m_dynamic_scalar_obs.clear();
  this->m_dynamic_vector_obs.clear();
  this->m_dynamic_matrix_obs.clear();

  this->m_eqtime_scalar_obs.shrink_to_fit();
  this->m_eqtime_vector_obs.shrink_to_fit();
  this->m_eqtime_matrix_obs.shrink_to_fit();
  this->m_dynamic_scalar_obs.shrink_to_fit();
  this->m_dynamic_vector_obs.shrink_to_fit();
  this->m_dynamic_matrix_obs.shrink_to_fit();

  if (this->m_equaltime_sign) {
    this->m_equaltime_sign.reset();
  }
  if (this->m_dynamic_sign) {
    this->m_dynamic_sign.reset();
  }
}

void ObservableHandler::initial(const ObsNameList& obs_list_in) {
  this->deallocate();

  ObsNameList obs_list = obs_list_in;
  std::sort(obs_list.begin(), obs_list.end());
  obs_list.erase(std::unique(obs_list.begin(), obs_list.end()), obs_list.end());

  if (!this->check_validity(obs_list)) {
    throw std::runtime_error(
        "Observable::ObservableHandler::initial(): "
        "unsupported observable type.");
  }

  for (const auto& obs_name : obs_list) {
    const auto& props = m_supported_observables.at(obs_name);

    ptrBaseObs obs_base = props.factory(obs_name, props.description);
    this->m_obs_map[obs_name] = obs_base;

    if (props.time_type == ObsTimeType::EqualTime) {
      switch (props.data_type) {
        case ObsDataType::Scalar:
          m_eqtime_scalar_obs.push_back(
              std::static_pointer_cast<ScalarObs>(obs_base));
          break;
        case ObsDataType::Vector:
          m_eqtime_vector_obs.push_back(
              std::static_pointer_cast<VectorObs>(obs_base));
          break;
        case ObsDataType::Matrix:
          m_eqtime_matrix_obs.push_back(
              std::static_pointer_cast<MatrixObs>(obs_base));
          break;
      }
    } else {  // ObsTimeType::Dynamic
      switch (props.data_type) {
        case ObsDataType::Scalar:
          m_dynamic_scalar_obs.push_back(
              std::static_pointer_cast<ScalarObs>(obs_base));
          break;
        case ObsDataType::Vector:
          m_dynamic_vector_obs.push_back(
              std::static_pointer_cast<VectorObs>(obs_base));
          break;
        case ObsDataType::Matrix:
          m_dynamic_matrix_obs.push_back(
              std::static_pointer_cast<MatrixObs>(obs_base));
          break;
      }
    }
  }

  // adding measurements of configuration signs manually to keep track of the
  // sign problem
  if (!this->m_eqtime_scalar_obs.empty() ||
      !this->m_eqtime_vector_obs.empty() ||
      !this->m_eqtime_matrix_obs.empty()) {
    ptrScalarObs equaltime_sign = std::make_shared<ScalarObs>(
        "equaltime_sign", "Averaged sign (equal-time)",
        Measure::Methods::measure_equaltime_config_sign);
    this->m_equaltime_sign = equaltime_sign;
    this->m_obs_map["equaltime_sign"] = equaltime_sign;
  }

  if (!this->m_dynamic_scalar_obs.empty() ||
      !this->m_dynamic_vector_obs.empty() ||
      !this->m_dynamic_matrix_obs.empty()) {
    ptrScalarObs dynamic_sign = std::make_shared<ScalarObs>(
        "dynamic_sign", "Averaged sign (dynamical)",
        Measure::Methods::measure_dynamic_config_sign);
    this->m_dynamic_sign = dynamic_sign;
    this->m_obs_map["dynamic_sign"] = dynamic_sign;
  }
}

}  // namespace Observable
