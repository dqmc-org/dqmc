#include "measure/observable_handler.h"

#include "measure/measure_methods.h"

namespace Observable {

namespace {
template <typename ObsType, typename MethodFunc>
auto make_observable(MethodFunc method) {
  return [method](const std::string& name,
                  const std::string& desc) -> std::unique_ptr<ObservableBase> {
    return std::make_unique<ObsType>(name, desc, method);
  };
}
}  // namespace

// clang-format off
const std::unordered_map<std::string, ObservableHandler::ObservableProperties> ObservableHandler::m_supported_observables = {
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
    {"pair_pair_corr_Q",               { "Pair–pair correlation at Q", ObsTimeType::EqualTime, ObsDataType::Scalar, make_observable<Scalar>(Measure::Methods::measure_pair_pair_corr_Q)}},
    // --- Dynamic Observables ---
    {"greens_functions",               {"Green's functions",           ObsTimeType::Dynamic,   ObsDataType::Matrix, make_observable<Matrix>(Measure::Methods::measure_greens_functions)}},
    {"density_of_states",              {"Density of states",           ObsTimeType::Dynamic,   ObsDataType::Vector, make_observable<Vector>(Measure::Methods::measure_density_of_states)}},
    {"superfluid_stiffness",           {"Superfluid stiffness",        ObsTimeType::Dynamic,   ObsDataType::Scalar, make_observable<Scalar>(Measure::Methods::measure_superfluid_stiffness)}},
    {"dynamic_spin_susceptibility",    {"Dynamic Spin Susceptibility", ObsTimeType::Dynamic,   ObsDataType::Vector, make_observable<Vector>(Measure::Methods::measure_dynamic_spin_susceptibility)}},
    {"dynamic_sign",                   {"Averaged sign (dynamical)",   ObsTimeType::Dynamic,   ObsDataType::Scalar, make_observable<Scalar>(Measure::Methods::measure_dynamic_config_sign)}},
    {"dynamic_pair_corr",              {"Dynamic pair correlator P(Q,tau)", ObsTimeType::Dynamic, ObsDataType::Vector, make_observable<Vector>(Measure::Methods::measure_dynamic_pair_corr)}},
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

bool ObservableHandler::is_eqtime(const std::string& obs_name) const {
  auto it = m_supported_observables.find(obs_name);
  return it != m_supported_observables.end() && it->second.time_type == ObsTimeType::EqualTime;
}

bool ObservableHandler::is_dynamic(const std::string& obs_name) const {
  auto it = m_supported_observables.find(obs_name);
  return it != m_supported_observables.end() && it->second.time_type == ObsTimeType::Dynamic;
}

bool ObservableHandler::check_validity(const std::vector<std::string>& obs_list) const {
  for (const auto& obs : obs_list) {
    if (m_supported_observables.find(obs) == m_supported_observables.end()) {
      return false;
    }
  }
  return true;
}

void ObservableHandler::initial(std::vector<std::string> obs_list) {
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
    throw std::runtime_error(dqmc_format_error("unsupported observable type."));
  }

  for (const auto& obs_name : obs_list) {
    const auto& props = m_supported_observables.at(obs_name);

    std::unique_ptr<ObservableBase> obs_ptr = props.factory(obs_name, props.description);

    // Non-owning raw pointer
    ObservableBase* raw_ptr = obs_ptr.get();

    // Transfer ownership
    this->m_obs_map[obs_name] = std::move(obs_ptr);

    auto& scalar_obs_list =
        (props.time_type == ObsTimeType::EqualTime) ? m_eqtime_scalar_obs : m_dynamic_scalar_obs;
    auto& vector_obs_list =
        (props.time_type == ObsTimeType::EqualTime) ? m_eqtime_vector_obs : m_dynamic_vector_obs;
    auto& matrix_obs_list =
        (props.time_type == ObsTimeType::EqualTime) ? m_eqtime_matrix_obs : m_dynamic_matrix_obs;

    switch (props.data_type) {
      case ObsDataType::Scalar:
        if (auto p = dynamic_cast<Scalar*>(raw_ptr)) {
          scalar_obs_list.push_back(p);
        }
        break;
      case ObsDataType::Vector:
        if (auto p = dynamic_cast<Vector*>(raw_ptr)) {
          vector_obs_list.push_back(p);
        }
        break;
      case ObsDataType::Matrix:
        if (auto p = dynamic_cast<Matrix*>(raw_ptr)) {
          matrix_obs_list.push_back(p);
        }
        break;
    }
  }
}

}  // namespace Observable
