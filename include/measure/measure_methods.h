#pragma once

/**
 *  This header file includes declarations of user-defined methods
 *  for the measurements of physical observables using dqmc.
 *  Both equal-time and dynamical measurements are supported.
 */

#include "measure/measure_context.h"
#include "measure/observable.h"

namespace Measure {

// forward declaration of MeasureHandler class
class MeasureHandler;

// --------------------------------------  Interface class Measure::Method
// --------------------------------------- provide user-defined measuring
// methods
namespace Methods {
// definitions of measuring methods
// arguments of method functions include Observable<ObsType> and MeasureContext.

// Equal-time Measurements:
//    1. Filling number < n >
//    2. Double occupancy D = < n_up * n_dn >
//    3. Single particle kinetic energy
//    4. Distributions of electrons in momentum space
//    5. Local spin correlation, magnetization C(0,0) = < ( n_up - n_dn )^2 >
//    6. Spin density structure factor (SDW)
//    7. Charge density structure factor (CDW)
//    8. S-wave Cooper pairing correlation functions

void measure_equaltime_config_sign(Observable::Scalar& equaltime_sign, const MeasureContext& ctx);

void measure_filling_number(Observable::Scalar& filling_number, const MeasureContext& ctx);

void measure_double_occupancy(Observable::Scalar& double_occupancy, const MeasureContext& ctx);

void measure_kinetic_energy(Observable::Scalar& kinetic_energy, const MeasureContext& ctx);

void measure_total_energy(Observable::Scalar& total_energy, const MeasureContext& ctx);

void measure_local_spin_corr(Observable::Scalar& local_spin_corr, const MeasureContext& ctx);

void measure_momentum_distribution(Observable::Scalar& momentum_dist, const MeasureContext& ctx);

void measure_spin_density_structure_factor(Observable::Scalar& sdw_factor,
                                           const MeasureContext& ctx);

void measure_charge_density_structure_factor(Observable::Scalar& cdw_factor,
                                             const MeasureContext& ctx);

void measure_s_wave_pairing_corr(Observable::Scalar& s_wave_pairing, const MeasureContext& ctx);

// Dynamical Measurements:
//    1. Dynamical green's functions in momentum space: G(k,t) = < c(k,t) *
//    c^+(k,0) >
//    2. Density of states in imaginary-time space: D(tau) = 1/N \sum i <
//    c(i,t) * c^+(i,0) >
//    3. Superfluid stiffness rho_s of superconducting: rho_s = ( Gamma_L -
//    Gamma_T ) / 4
//    4. (local) Dynamic spin susceptibility, proportional to 1/T1 from NMR
//    experiments, 1/T1 = \sum q < Sz(q,t) Sz(q,0) >

void measure_dynamic_config_sign(Observable::Scalar& dynamic_sign, const MeasureContext& ctx);

void measure_greens_functions(Observable::Matrix& greens_functions, const MeasureContext& ctx);

void measure_density_of_states(Observable::Vector& density_of_states, const MeasureContext& ctx);

void measure_superfluid_stiffness(Observable::Scalar& superfluid_stiffness,
                                  const MeasureContext& ctx);

void measure_dynamic_spin_susceptibility(Observable::Vector& dynamic_spin_susceptibility,
                                         const MeasureContext& ctx);

void measure_pair_pair_corr_Q(Observable::Scalar& pair_corr_Q, const MeasureContext& ctx);

void measure_dynamic_pair_corr(Observable::Vector& dynamic_pair_corr, const MeasureContext& ctx);
}  // namespace Methods
}  // namespace Measure
