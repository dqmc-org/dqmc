#pragma once

/**
 *  This header file includes declarations of user-defined methods
 *  for the measurements of physical observables using dqmc.
 *  Both equal-time and dynamical measurements are supported.
 */

#include "measure/observable.h"

// forward declaration
namespace Model {
class ModelBase;
}
namespace Lattice {
class LatticeBase;
}
namespace DQMC {
class Walker;
}

namespace Measure {

// forward declaration of MeasureHandler class
class MeasureHandler;

using ModelBase = Model::ModelBase;
using LatticeBase = Lattice::LatticeBase;
using Walker = DQMC::Walker;

// --------------------------------------  Interface class Measure::Method
// --------------------------------------- provide user-defined measuring
// methods
class Methods {
 public:
  // definitions of measuring methods
  // arguments of method functions should include Observable<ObsType>,
  // Measure::MeasureHandler, DQMC::Walker,
  // Model::ModelBase and Lattice::LatticeBase class.

  // Equal-time Measurements:
  //    1. Filling number < n >
  //    2. Double occupancy D = < n_up * n_dn >
  //    3. Single particle kinetic energy
  //    4. Distributions of electrons in momentum space
  //    5. Local spin correlation, magnetization C(0,0) = < ( n_up - n_dn )^2 >
  //    6. Spin density structure factor (SDW)
  //    7. Charge density structure factor (CDW)
  //    8. S-wave Cooper pairing correlation functions

  static void measure_equaltime_config_sign(Observable::Scalar& equaltime_sign,
                                            const MeasureHandler& meas_handler,
                                            const Walker& walker, const ModelBase& model,
                                            const LatticeBase& lattice);

  static void measure_filling_number(Observable::Scalar& filling_number,
                                     const MeasureHandler& meas_handler, const Walker& walker,
                                     const ModelBase& model, const LatticeBase& lattice);

  static void measure_double_occupancy(Observable::Scalar& double_occupancy,
                                       const MeasureHandler& meas_handler, const Walker& walker,
                                       const ModelBase& model, const LatticeBase& lattice);

  static void measure_kinetic_energy(Observable::Scalar& kinetic_energy,
                                     const MeasureHandler& meas_handler, const Walker& walker,
                                     const ModelBase& model, const LatticeBase& lattice);

  static void measure_local_spin_corr(Observable::Scalar& local_spin_corr,
                                      const MeasureHandler& meas_handler, const Walker& walker,
                                      const ModelBase& model, const LatticeBase& lattice);

  static void measure_momentum_distribution(Observable::Scalar& momentum_dist,
                                            const MeasureHandler& meas_handler,
                                            const Walker& walker, const ModelBase& model,
                                            const LatticeBase& lattice);

  static void measure_spin_density_structure_factor(Observable::Scalar& sdw_factor,
                                                    const MeasureHandler& meas_handler,
                                                    const Walker& walker, const ModelBase& model,
                                                    const LatticeBase& lattice);

  static void measure_charge_density_structure_factor(Observable::Scalar& cdw_factor,
                                                      const MeasureHandler& meas_handler,
                                                      const Walker& walker, const ModelBase& model,
                                                      const LatticeBase& lattice);

  static void measure_s_wave_pairing_corr(Observable::Scalar& s_wave_pairing,
                                          const MeasureHandler& meas_handler, const Walker& walker,
                                          const ModelBase& model, const LatticeBase& lattice);

  // Dynamical Measurements:
  //    1. Dynamical green's functions in momentum space: G(k,t) = < c(k,t) *
  //    c^+(k,0) >
  //    2. Density of states in imaginary-time space: D(tau) = 1/N \sum i <
  //    c(i,t) * c^+(i,0) >
  //    3. Superfluid stiffness rho_s of superconducting: rho_s = ( Gamma_L -
  //    Gamma_T ) / 4
  //    4. (local) Dynamic spin susceptibility, proportional to 1/T1 from NMR
  //    experiments, 1/T1 = \sum q < Sz(q,t) Sz(q,0) >

  static void measure_dynamic_config_sign(Observable::Scalar& dynamic_sign,
                                          const MeasureHandler& meas_handler, const Walker& walker,
                                          const ModelBase& model, const LatticeBase& lattice);

  static void measure_greens_functions(Observable::Matrix& greens_functions,
                                       const MeasureHandler& meas_handler, const Walker& walker,
                                       const ModelBase& model, const LatticeBase& lattice);

  static void measure_density_of_states(Observable::Vector& density_of_states,
                                        const MeasureHandler& meas_handler, const Walker& walker,
                                        const ModelBase& model, const LatticeBase& lattice);

  static void measure_superfluid_stiffness(Observable::Scalar& superfluid_stiffness,
                                           const MeasureHandler& meas_handler, const Walker& walker,
                                           const ModelBase& model, const LatticeBase& lattice);

  static void measure_dynamic_spin_susceptibility(Observable::Vector& dynamic_spin_susceptibility,
                                                  const MeasureHandler& meas_handler,
                                                  const Walker& walker, const ModelBase& model,
                                                  const LatticeBase& lattice);
};
}  // namespace Measure
