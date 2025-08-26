#include "measure/observable_handler.h"

#include <format>
#include <iostream>

#include "measure/measure_methods.h"

namespace Observable {

namespace {
template <typename ObsType, typename MethodFunc>
auto make_observable(MethodFunc method) {
  return [method](const ObsName& name, const std::string& desc) -> std::shared_ptr<ObservableBase> {
    return std::make_shared<ObsType>(name, desc, method);
  };
}
}  // namespace

// clang-format off
const std::map<std::string, ObservableHandler::ObservableProperties> ObservableHandler::m_supported_observables = {
    // --- Equal-Time Observables ---
    {"filling_number",                 {"Filling number",              ObsTimeType::EqualTime, ObsDataType::Scalar, make_observable<Scalar>(Measure::Methods::measure_filling_number)}},
    {"double_occupancy",               {"Double occupation",           ObsTimeType::EqualTime, ObsDataType::Scalar, make_observable<Scalar>(Measure::Methods::measure_double_occupancy)}},
    {"kinetic_energy",                 {"Kinetic energy",              ObsTimeType::EqualTime, ObsDataType::Scalar, make_observable<Scalar>(Measure::Methods::measure_kinetic_energy)}},
    {"momentum_distribution",          {"Momentum distribution",       ObsTimeType::EqualTime, ObsDataType::Scalar, make_observable<Scalar>(Measure::Methods::measure_momentum_distribution)}},
    {"local_spin_corr",                {"Local spin correlation",      ObsTimeType::EqualTime, ObsDataType::Scalar, make_observable<Scalar>(Measure::Methods::measure_local_spin_corr)}},
    {"spin_density_structure_factor",  {"SDW order parameter",         ObsTimeType::EqualTime, ObsDataType::Scalar, make_observable<Scalar>(Measure::Methods::measure_spin_density_structure_factor)}},
    {"charge_density_structure_factor",{"CDW order parameter",         ObsTimeType::EqualTime, ObsDataType::Scalar, make_observable<Scalar>(Measure::Methods::measure_charge_density_structure_factor)}},
    {"s_wave_pairing_corr",            {"S-wave pairing correlation",  ObsTimeType::EqualTime, ObsDataType::Scalar, make_observable<Scalar>(Measure::Methods::measure_s_wave_pairing_corr)}},
    {"equaltime_sign",                 {"Averaged sign (equal-time)",  ObsTimeType::EqualTime, ObsDataType::Scalar, make_observable<Scalar>(Measure::Methods::measure_equaltime_config_sign)}},

    // --- Dynamic Observables ---
    {"greens_functions",               {"Green's functions",           ObsTimeType::Dynamic,   ObsDataType::Matrix, make_observable<Matrix>(Measure::Methods::measure_greens_functions)}},
    {"density_of_states",              {"Density of states",           ObsTimeType::Dynamic,   ObsDataType::Vector, make_observable<Vector>(Measure::Methods::measure_density_of_states)}},
    {"superfluid_stiffness",           {"Superfluid stiffness",        ObsTimeType::Dynamic,   ObsDataType::Scalar, make_observable<Scalar>(Measure::Methods::measure_superfluid_stiffness)}},
    {"dynamic_spin_susceptibility",    {"Dynamic Spin Susceptibility", ObsTimeType::Dynamic,   ObsDataType::Vector, make_observable<Vector>(Measure::Methods::measure_dynamic_spin_susceptibility)}},
    {"dynamic_sign",                   {"Averaged sign (dynamical)",   ObsTimeType::Dynamic,   ObsDataType::Scalar, make_observable<Scalar>(Measure::Methods::measure_dynamic_config_sign)}},
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
  return it != m_supported_observables.end() && it->second.time_type == ObsTimeType::EqualTime;
}

bool ObservableHandler::is_dynamic(const ObsName& obs_name) const {
  auto it = m_supported_observables.find(obs_name);
  return it != m_supported_observables.end() && it->second.time_type == ObsTimeType::Dynamic;
}

bool ObservableHandler::check_validity(const ObsNameList& obs_list) const {
  for (const auto& obs : obs_list) {
    if (m_supported_observables.find(obs) == m_supported_observables.end()) {
      std::string error_message = std::format("Error: observable '{}' is not supported.", obs);
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
}

void ObservableHandler::initial(const ObsNameList& obs_list_in) {
  this->deallocate();

  ObsNameList obs_list = obs_list_in;

  // Automatically add sign observables if there are equal-time or dynamic observables
  bool has_eqtime = false;
  bool has_dynamic = false;

  for (const auto& obs_name : obs_list) {
    if (this->is_eqtime(obs_name)) {
      has_eqtime = true;
    } else if (this->is_dynamic(obs_name)) {
      has_dynamic = true;
    }
  }

  if (has_eqtime) {
    obs_list.push_back("equaltime_sign");
  }

  if (has_dynamic) {
    obs_list.push_back("dynamic_sign");
  }

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
          m_eqtime_scalar_obs.push_back(std::static_pointer_cast<Scalar>(obs_base));
          break;
        case ObsDataType::Vector:
          m_eqtime_vector_obs.push_back(std::static_pointer_cast<Vector>(obs_base));
          break;
        case ObsDataType::Matrix:
          m_eqtime_matrix_obs.push_back(std::static_pointer_cast<Matrix>(obs_base));
          break;
      }
    } else {  // ObsTimeType::Dynamic
      switch (props.data_type) {
        case ObsDataType::Scalar:
          m_dynamic_scalar_obs.push_back(std::static_pointer_cast<Scalar>(obs_base));
          break;
        case ObsDataType::Vector:
          m_dynamic_vector_obs.push_back(std::static_pointer_cast<Vector>(obs_base));
          break;
        case ObsDataType::Matrix:
          m_dynamic_matrix_obs.push_back(std::static_pointer_cast<Matrix>(obs_base));
          break;
      }
    }
  }
}

}  // namespace Observable
