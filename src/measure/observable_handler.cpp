#include "measure/observable_handler.h"

#include <iostream>

#include "measure/measure_methods.h"

namespace Observable {

// definitions of static members
// tables of all supported physical observables for measurements
// equal-time observables
ObsTable ObservableHandler::m_eqtime_obs_table = {
    "filling_number",
    "double_occupancy",
    "kinetic_energy",
    "momentum_distribution",
    "local_spin_corr",
    "spin_density_structure_factor",
    "charge_density_structure_factor",
    "s_wave_pairing_corr",
};

// dynamic observables
ObsTable ObservableHandler::m_dynamic_obs_table = {
    "greens_functions",
    "density_of_states",
    "superfluid_stiffness",
    "dynamic_spin_susceptibility",
};

// public member for external calls
ObsTable ObservableHandler::ObservableAll = {
    "filling_number",
    "double_occupancy",
    "kinetic_energy",
    "momentum_distribution",
    "local_spin_corr",
    "spin_density_structure_factor",
    "charge_density_structure_factor",
    "s_wave_pairing_corr",
    "greens_functions",
    "density_of_states",
    "superfluid_stiffness",
    "dynamic_spin_susceptibility",
};

bool ObservableHandler::is_eqtime(const ObsName& obs_name) const {
  return (std::find(this->m_eqtime_obs_table.begin(),
                    this->m_eqtime_obs_table.end(),
                    obs_name) != this->m_eqtime_obs_table.end());
}

bool ObservableHandler::is_dynamic(const ObsName& obs_name) const {
  return (std::find(this->m_dynamic_obs_table.begin(),
                    this->m_dynamic_obs_table.end(),
                    obs_name) != this->m_dynamic_obs_table.end());
}

bool ObservableHandler::find(const ObsName& obs_name) {
  return (this->m_obs_map.find(obs_name) != this->m_obs_map.end());
}

bool ObservableHandler::check_validity(const ObsNameList& obs_list) const {
  // preprocessing
  // remove redundant input
  ObsNameList tmp_list = obs_list;
  std::sort(tmp_list.begin(), tmp_list.end());
  tmp_list.erase(unique(tmp_list.begin(), tmp_list.end()), tmp_list.end());

  // check the validity of the input
  for (const auto& obs : tmp_list) {
    if (!this->is_eqtime(obs) && !this->is_dynamic(obs)) {
      return false;
    }
  }
  // otherwise return true
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

void ObservableHandler::initial(const ObsNameList& obs_list) {
  // release memory if previously initialized
  this->deallocate();

  // check the validity of the input
  if (!this->check_validity(obs_list)) {
    // unsupported observables found, throw errors
    std::cerr << "Observable::ObservableHandler::initial(): "
              << "unsupported observable type from the input." << std::endl;
    exit(1);
  }

  for (const auto& obs_name : obs_list) {
    // allocate memory for observables
    // caution that to this stage only properties like name or method is
    // assigned. other info, such as m_size_of_bin and dimensional of
    // m_zero_elem, is kept unassigned until MeasureHandler or Lattice class is
    // specialized.

    // -----------------------------------------------------------------------------------------------
    //                                    Equal-time Observables
    // -----------------------------------------------------------------------------------------------

    // -------------------------------------  Filling number
    // ----------------------------------------
    if (obs_name == "filling_number") {
      ptrScalarObs filling_number =
          std::make_shared<ScalarObs>(obs_name, "Filling number");
      filling_number->add_method(Measure::Methods::measure_filling_number);
      this->m_eqtime_scalar_obs.emplace_back(filling_number);

      // fill in the map
      this->m_obs_map[obs_name] =
          std::static_pointer_cast<ObservableBase>(filling_number);
    }

    // ------------------------------------  Double occupancy
    // ---------------------------------------
    if (obs_name == "double_occupancy") {
      ptrScalarObs double_occupancy =
          std::make_shared<ScalarObs>(obs_name, "Double occupation");
      double_occupancy->add_method(Measure::Methods::measure_double_occupancy);
      this->m_eqtime_scalar_obs.emplace_back(double_occupancy);
      this->m_obs_map[obs_name] =
          std::static_pointer_cast<ObservableBase>(double_occupancy);
    }

    // ------------------------------------  Kinetic energy
    // -----------------------------------------
    if (obs_name == "kinetic_energy") {
      ptrScalarObs kinetic_energy =
          std::make_shared<ScalarObs>(obs_name, "Kinetic energy");
      kinetic_energy->add_method(Measure::Methods::measure_kinetic_energy);
      this->m_eqtime_scalar_obs.emplace_back(kinetic_energy);
      this->m_obs_map[obs_name] =
          std::static_pointer_cast<ObservableBase>(kinetic_energy);
    }

    // ---------------------------------  Momentum distribution
    // -------------------------------------
    if (obs_name == "momentum_distribution") {
      ptrScalarObs momentum_distribution =
          std::make_shared<ScalarObs>(obs_name, "Momentum distribution");
      momentum_distribution->add_method(
          Measure::Methods::measure_momentum_distribution);
      this->m_eqtime_scalar_obs.emplace_back(momentum_distribution);
      this->m_obs_map[obs_name] =
          std::static_pointer_cast<ObservableBase>(momentum_distribution);
    }

    // --------------------------------  Local spin correlations
    // ------------------------------------
    if (obs_name == "local_spin_corr") {
      ptrScalarObs local_spin_corr =
          std::make_shared<ScalarObs>(obs_name, "Local spin correlation");
      local_spin_corr->add_method(Measure::Methods::measure_local_spin_corr);
      this->m_eqtime_scalar_obs.emplace_back(local_spin_corr);
      this->m_obs_map[obs_name] =
          std::static_pointer_cast<ObservableBase>(local_spin_corr);
    }

    // -------------------------  Spin density wave (SDW) structure factor
    // --------------------------
    if (obs_name == "spin_density_structure_factor") {
      ptrScalarObs sdw_factor =
          std::make_shared<ScalarObs>(obs_name, "SDW order parameter");
      sdw_factor->add_method(
          Measure::Methods::measure_spin_density_structure_factor);
      this->m_eqtime_scalar_obs.emplace_back(sdw_factor);
      this->m_obs_map[obs_name] =
          std::static_pointer_cast<ObservableBase>(sdw_factor);
    }

    // ------------------------  Charge density wave (CDW) structure factor
    // -------------------------
    if (obs_name == "charge_density_structure_factor") {
      ptrScalarObs cdw_factor =
          std::make_shared<ScalarObs>(obs_name, "CDW order parameter");
      cdw_factor->add_method(
          Measure::Methods::measure_charge_density_structure_factor);
      this->m_eqtime_scalar_obs.emplace_back(cdw_factor);
      this->m_obs_map[obs_name] =
          std::static_pointer_cast<ObservableBase>(cdw_factor);
    }

    // ----------------------  S wave pairing correlations of superconductivity
    // ---------------------
    if (obs_name == "s_wave_pairing_corr") {
      ptrScalarObs s_wave_pairing_corr =
          std::make_shared<ScalarObs>(obs_name, "S-wave pairing correlation");
      s_wave_pairing_corr->add_method(
          Measure::Methods::measure_s_wave_pairing_corr);
      this->m_eqtime_scalar_obs.emplace_back(s_wave_pairing_corr);
      this->m_obs_map[obs_name] =
          std::static_pointer_cast<ObservableBase>(s_wave_pairing_corr);
    }

    // adding new methods here

    // -----------------------------------------------------------------------------------------------
    //                                  Time-displaced Observables
    // -----------------------------------------------------------------------------------------------

    // -------------------------------  Greens functions G(k, tau)
    // ----------------------------------
    if (obs_name == "greens_functions") {
      ptrMatrixObs greens_functions =
          std::make_shared<MatrixObs>(obs_name, "Green's functions");
      greens_functions->add_method(Measure::Methods::measure_greens_functions);
      this->m_dynamic_matrix_obs.emplace_back(greens_functions);
      this->m_obs_map[obs_name] =
          std::static_pointer_cast<ObservableBase>(greens_functions);
    }

    // --------------------------------  Density of states D(tau)
    // -----------------------------------
    if (obs_name == "density_of_states") {
      ptrVectorObs density_of_states =
          std::make_shared<VectorObs>(obs_name, "Density of states");
      density_of_states->add_method(
          Measure::Methods::measure_density_of_states);
      this->m_dynamic_vector_obs.emplace_back(density_of_states);
      this->m_obs_map[obs_name] =
          std::static_pointer_cast<ObservableBase>(density_of_states);
    }

    // ---------------------------------  Superfluid stiffness
    // --------------------------------------
    if (obs_name == "superfluid_stiffness") {
      ptrScalarObs superfluid_stiffness =
          std::make_shared<ScalarObs>(obs_name, "Superfluid stiffness");
      superfluid_stiffness->add_method(
          Measure::Methods::measure_superfluid_stiffness);
      this->m_dynamic_scalar_obs.emplace_back(superfluid_stiffness);
      this->m_obs_map[obs_name] =
          std::static_pointer_cast<ObservableBase>(superfluid_stiffness);
    }

    // -------------------------------  Dynamic spin susceptibility
    // ---------------------------------
    if (obs_name == "dynamic_spin_susceptibility") {
      ptrVectorObs dynamic_spin_susceptibility =
          std::make_shared<VectorObs>(obs_name, "Dynamic Spin Susceptibility");
      dynamic_spin_susceptibility->add_method(
          Measure::Methods::measure_dynamic_spin_susceptibility);
      this->m_dynamic_vector_obs.emplace_back(dynamic_spin_susceptibility);
      this->m_obs_map[obs_name] =
          std::static_pointer_cast<ObservableBase>(dynamic_spin_susceptibility);
    }

    // add new methods here
  }

  // adding measurements of configuration signs manually to keep track of the
  // sign problem
  if (!this->m_eqtime_scalar_obs.empty() ||
      !this->m_eqtime_vector_obs.empty() ||
      !this->m_eqtime_matrix_obs.empty()) {
    ptrScalarObs equaltime_sign = std::make_shared<ScalarObs>(
        "equaltime_sign", "Averaged sign (equal-time)");
    equaltime_sign->add_method(Measure::Methods::measure_equaltime_config_sign);
    this->m_equaltime_sign = equaltime_sign;
    this->m_obs_map["equaltime_sign"] =
        std::static_pointer_cast<ObservableBase>(equaltime_sign);
  }

  if (!this->m_dynamic_scalar_obs.empty() ||
      !this->m_dynamic_vector_obs.empty() ||
      !this->m_dynamic_matrix_obs.empty()) {
    ptrScalarObs dynamic_sign = std::make_shared<ScalarObs>(
        "dynamic_sign", "Averaged sign (dynamical)");
    dynamic_sign->add_method(Measure::Methods::measure_dynamic_config_sign);
    this->m_dynamic_sign = dynamic_sign;
    this->m_obs_map["dynamic_sign"] =
        std::static_pointer_cast<ObservableBase>(dynamic_sign);
  }
}

}  // namespace Observable
