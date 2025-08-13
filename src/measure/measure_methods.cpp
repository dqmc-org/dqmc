#include "measure/measure_methods.h"

#include "lattice/lattice_base.h"
#include "lattice/square.h"
#include "measure/measure_handler.h"
#include "model/model_base.h"
#include "walker.h"

namespace Measure {

// some aliases
using RealScalar = double;
using GreensFunc = Eigen::MatrixXd;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

// -----------------------------  Method routines for equal-time measurements
// -------------------------------

// The sign of the bosonic field configurations for equal-time measurements,
// useful for reweighting
void Methods::measure_equaltime_config_sign(ScalarObs& equaltime_sign,
                                            const MeasureHandler& meas_handler,
                                            const Walker& walker,
                                            const ModelBase& model,
                                            const LatticeBase& lattice) {
  equaltime_sign.tmp_value() += walker.vecConfigSign().sum();
  equaltime_sign.counts() += walker.TimeSize();
}

// Filling number defined as \sum i ( n_up + n_dn )(i)
// which represents the total number of electrons
void Methods::measure_filling_number(ScalarObs& filling_number,
                                     const MeasureHandler& meas_handler,
                                     const Walker& walker,
                                     const ModelBase& model,
                                     const LatticeBase& lattice) {
  // loop over equivalent time slices
  for (auto t = 0; t < walker.TimeSize(); ++t) {
    filling_number.tmp_value() +=
        walker.ConfigSign(t) *
        (2 - (walker.GreenttUp(t).trace() + walker.GreenttDn(t).trace()) /
                 lattice.SpaceSize());
    ++filling_number;
  }
}

// Double occupation defined as \sum i ( n_up * n_dn )(i)
// which quantizes the possibility that two electrons with opposite spin occupy
// the same site
void Methods::measure_double_occupancy(ScalarObs& double_occupancy,
                                       const MeasureHandler& meas_handler,
                                       const Walker& walker,
                                       const ModelBase& model,
                                       const LatticeBase& lattice) {
  for (auto t = 0; t < walker.TimeSize(); ++t) {
    const GreensFunc& gu = walker.GreenttUp(t);
    const GreensFunc& gd = walker.GreenttDn(t);
    const RealScalar& config_sign = walker.ConfigSign(t);

    RealScalar tmp_double_occu = 0.0;
    for (int i = 0; i < lattice.SpaceSize(); ++i) {
      tmp_double_occu += (1 - gu(i, i)) * (1 - gd(i, i));
    }
    double_occupancy.tmp_value() +=
        config_sign * tmp_double_occu / lattice.SpaceSize();
    ++double_occupancy;
  }
}

// Kinetic energy defined as -t \sum <ij> ( c^+_j c_i + h.c. )
// which measures the hoppings between two sites, e.g. hoppings between nearest
// neighbours
void Methods::measure_kinetic_energy(ScalarObs& kinetic_energy,
                                     const MeasureHandler& meas_handler,
                                     const Walker& walker,
                                     const ModelBase& model,
                                     const LatticeBase& lattice) {
  for (auto t = 0; t < walker.TimeSize(); ++t) {
    const GreensFunc& gu = walker.GreenttUp(t);
    const GreensFunc& gd = walker.GreenttDn(t);
    const RealScalar& config_sign = walker.ConfigSign(t);

    RealScalar tmp_kinetic_energy = 0.0;
    for (auto i = 0; i < lattice.SpaceSize(); ++i) {
      // loop over lattice sites
      for (auto dir = 0; dir < lattice.SpaceDim(); ++dir) {
        // loop over nearest neighbours
        // todo: the independent directions should equal to the coordination
        // number of the lattice
        tmp_kinetic_energy += gu(i, lattice.NearestNeighbour(i, dir)) +
                              gd(i, lattice.NearestNeighbour(i, dir));
      }
    }
    kinetic_energy.tmp_value() += config_sign * (2 * model.HoppingT()) *
                                  tmp_kinetic_energy / lattice.SpaceSize();
    ++kinetic_energy;
  }
}

// In general, spin correlations is defined as C(i,t) = < (n_up - n_dn)(i,t) *
// (n_up - n_dn)(0,0) > which measure the correlations of spins between two
// space-time points. the local correlations are the limit of i = 0 and t = 0.
void Methods::measure_local_spin_corr(ScalarObs& local_spin_corr,
                                      const MeasureHandler& meas_handler,
                                      const Walker& walker,
                                      const ModelBase& model,
                                      const LatticeBase& lattice) {
  for (auto t = 0; t < walker.TimeSize(); ++t) {
    const GreensFunc& gu = walker.GreenttUp(t);
    const GreensFunc& gd = walker.GreenttDn(t);
    const RealScalar& config_sign = walker.ConfigSign(t);

    RealScalar tmp_local_spin_corr = 0.0;
    for (auto i = 0; i < lattice.SpaceSize(); ++i) {
      tmp_local_spin_corr +=
          config_sign * (gu(i, i) + gd(i, i) - 2 * gu(i, i) * gd(i, i));
    }
    local_spin_corr.tmp_value() += tmp_local_spin_corr / lattice.SpaceSize();
    ++local_spin_corr;
  }
}

// Distribution of electrons in momentum space defined as n(k) = ( n_up + n_dn
// )(k) measured for one specific momentum point todo: scan the momentum space
void Methods::measure_momentum_distribution(ScalarObs& momentum_dist,
                                            const MeasureHandler& meas_handler,
                                            const Walker& walker,
                                            const ModelBase& model,
                                            const LatticeBase& lattice) {
  for (auto t = 0; t < walker.TimeSize(); ++t) {
    const GreensFunc& gu = walker.GreenttUp(t);
    const GreensFunc& gd = walker.GreenttDn(t);
    const RealScalar& config_sign = walker.ConfigSign(t);

    RealScalar tmp_momentum_dist = 0.0;
    // the first site i
    for (auto i = 0; i < lattice.SpaceSize(); ++i) {
      // the second site j
      for (auto j = 0; j < lattice.SpaceSize(); ++j) {
        tmp_momentum_dist += (gu(j, i) + gd(j, i)) *
                             lattice.FourierFactor(lattice.Displacement(i, j),
                                                   meas_handler.Momentum());
      }
    }
    momentum_dist.tmp_value() +=
        config_sign * (1 - 0.5 * tmp_momentum_dist / lattice.SpaceSize());
    ++momentum_dist;
  }
}

// Structure factor of spin density wave (SDW) defined as
// 1/N \sum ij ( exp( -i Q*(ri-rj) ) * (n_up - n_dn)(j) * (n_up - n_dn)(i) )
// where Q is the wave momentum of sdw.
void Methods::measure_spin_density_structure_factor(
    ScalarObs& sdw_factor, const MeasureHandler& meas_handler,
    const Walker& walker, const ModelBase& model, const LatticeBase& lattice) {
  const int space_size = lattice.SpaceSize();
  const double inv_space_size_sq =
      1.0 / (static_cast<double>(space_size) * space_size);
  const int K_vector = meas_handler.Momentum();

  for (auto t = 0; t < walker.TimeSize(); ++t) {
    const GreensFunc& gu = walker.GreenttUp(t);
    const GreensFunc& gd = walker.GreenttDn(t);
    const GreensFunc& guc =
        Matrix::Identity(space_size, space_size) - gu.transpose();
    const GreensFunc& gdc =
        Matrix::Identity(space_size, space_size) - gd.transpose();
    const double config_sign = walker.ConfigSign(t);

    RealScalar tmp_sdw = 0.0;
    for (auto i = 0; i < space_size; ++i) {
      const double guc_ii = guc(i, i);
      const double gdc_ii = gdc(i, i);

      for (auto j = 0; j < space_size; ++j) {
        tmp_sdw +=
            lattice.FourierFactor(lattice.Displacement(i, j), K_vector) *
            (+guc_ii * guc(j, j) + guc(i, j) * gu(i, j) + gdc_ii * gdc(j, j) +
             gdc(i, j) * gd(i, j) - gdc_ii * guc(j, j) - guc_ii * gdc(j, j));
      }
    }

    sdw_factor.tmp_value() += config_sign * tmp_sdw * inv_space_size_sq;
    ++sdw_factor;
  }
}
// Structure factor of charge density wave (CDW) defined as
// 1/N \sum ij ( exp( -i Q*(ri-rj) ) * (n_up + n_dn)(j) * (n_up + n_dn)(i) )
// where Q is the wave momentum of cdw.
void Methods::measure_charge_density_structure_factor(
    ScalarObs& cdw_factor, const MeasureHandler& meas_handler,
    const Walker& walker, const ModelBase& model, const LatticeBase& lattice) {
  const int space_size = lattice.SpaceSize();
  const int time_size = walker.TimeSize();
  const int K_vector = meas_handler.Momentum();
  const double inv_space_size_sq =
      1.0 / (static_cast<double>(space_size) * space_size);

  std::vector<double> n_up(space_size);
  std::vector<double> n_dn(space_size);

  for (auto t = 0; t < time_size; ++t) {
    const GreensFunc& gu = walker.GreenttUp(t);
    const GreensFunc& gd = walker.GreenttDn(t);
    const double config_sign = walker.ConfigSign(t);

    for (int i = 0; i < space_size; ++i) {
      n_up[i] = 1.0 - gu(i, i);
      n_dn[i] = 1.0 - gd(i, i);
    }

    double tmp_cdw = 0.0;
    for (auto i = 0; i < space_size; ++i) {
      const double ni = n_up[i] + n_dn[i];

      for (auto j = 0; j < space_size; ++j) {
        const double nj = n_up[j] + n_dn[j];
        const double density_term = ni * nj;

        const double guc_ij = (i == j) - gu(j, i);
        const double gdc_ij = (i == j) - gd(j, i);
        const double hopping_term = guc_ij * gu(i, j) + gdc_ij * gd(i, j);

        const double total_correlator = density_term + hopping_term;
        const auto fourier_factor =
            lattice.FourierFactor(lattice.Displacement(i, j), K_vector);

        tmp_cdw += fourier_factor * total_correlator;
      }
    }
    cdw_factor.tmp_value() += config_sign * tmp_cdw * inv_space_size_sq;
    ++cdw_factor;
  }
}

// The s-wave superconducting pairing is defined as Delta = 1/sqrt(N) \sum i (
// c_up * c_dn )(i) Accordingly, the correlation function Ps reads
//   Ps  =  1/2 ( Delta^+ * Delta + h.c. )
//       =  1/N \sum ij ( (delta_ij - Gup(j,i)) * (delta_ij - Gdn(j,i)) )
// which serves as the Laudau order paramerter, and, with special attention, is
// an extensive quantity. Note the 1/2 prefactor in definition of Ps cancels the
// duplicated countings of ij.
void Methods::measure_s_wave_pairing_corr(ScalarObs& s_wave_pairing,
                                          const MeasureHandler& meas_handler,
                                          const Walker& walker,
                                          const ModelBase& model,
                                          const LatticeBase& lattice) {
  for (auto t = 0; t < walker.TimeSize(); ++t) {
    //  g(i,j) = < c_i * c^+_j > are the greens functions
    // gc(i,j) = < c^+_i * c_j > are isomorphic to the conjugation of greens
    // functions
    const GreensFunc& gu = walker.GreenttUp(t);
    const GreensFunc& gd = walker.GreenttDn(t);
    const GreensFunc& guc =
        Matrix::Identity(lattice.SpaceSize(), lattice.SpaceSize()) -
        gu.transpose();
    const GreensFunc& gdc =
        Matrix::Identity(lattice.SpaceSize(), lattice.SpaceSize()) -
        gd.transpose();
    const RealScalar& config_sign = walker.ConfigSign(t);

    // loop over site i, j and take averages
    RealScalar tmp_s_wave_pairing = 0.0;
    for (auto i = 0; i < lattice.SpaceSize(); ++i) {
      for (auto j = 0; j < lattice.SpaceSize(); ++j) {
        tmp_s_wave_pairing += config_sign * (guc(i, j) * gdc(i, j));
      }
    }
    // entensive quantity
    s_wave_pairing.tmp_value() += tmp_s_wave_pairing / lattice.SpaceSize();
    ++s_wave_pairing;
  }
}

// ------------------------------  Method routines for dynamic measurements
// ---------------------------------

// The sign of the bosonic field configurations for dynamic measurements
void Methods::measure_dynamic_config_sign(ScalarObs& dynamic_sign,
                                          const MeasureHandler& meas_handler,
                                          const Walker& walker,
                                          const ModelBase& model,
                                          const LatticeBase& lattice) {
  dynamic_sign.tmp_value() += walker.ConfigSign();
  ++dynamic_sign;
}

// Green's functions G(k,t) = < c(k,t) c^+(k,0) > in momentum space
// which are defined as the Fourier transmations of G(i,j) in real space
// G(k,t)  =  1/N \sum ij exp( -i k*(rj-ri) ) * ( c_j(t) * c^+_i(0) )
void Methods::measure_greens_functions(MatrixObs& greens_functions,
                                       const MeasureHandler& meas_handler,
                                       const Walker& walker,
                                       const ModelBase& model,
                                       const LatticeBase& lattice) {
  const int time_size = walker.TimeSize();
  const int space_size = lattice.SpaceSize();
  const int num_momenta = meas_handler.MomentumList().size();
  const double config_sign = walker.ConfigSign();
  const double prefactor = config_sign / static_cast<double>(space_size);

  for (auto t = 0; t < time_size; ++t) {
    const int tau = (t == 0) ? time_size - 1 : t - 1;

    const GreensFunc& gup =
        (t == 0) ? walker.GreenttUp(tau) : walker.Greent0Up(tau);
    const GreensFunc& gdn =
        (t == 0) ? walker.GreenttDn(tau) : walker.Greent0Dn(tau);

    for (auto k = 0; k < num_momenta; ++k) {
      const auto& K_vector = meas_handler.MomentumList(k);

      double current_k_t_sum = 0.0;

      for (auto i = 0; i < space_size; ++i) {
        for (auto j = 0; j < space_size; ++j) {
          const double gt0_ji = 0.5 * (gup(j, i) + gdn(j, i));

          const auto fourier_factor =
              lattice.FourierFactor(lattice.Displacement(i, j), K_vector);

          current_k_t_sum += gt0_ji * fourier_factor;
        }
      }
      greens_functions.tmp_value()(k, t) += prefactor * current_k_t_sum;
    }
  }
  ++greens_functions;
}

// Density of states D(t) defined as 1/N \sum i ( c(i,t) * c^+(i,0) )
// whose fourier transformations are exactly the usual density of states
// D(omega).
void Methods::measure_density_of_states(VectorObs& density_of_states,
                                        const MeasureHandler& meas_handler,
                                        const Walker& walker,
                                        const ModelBase& model,
                                        const LatticeBase& lattice) {
  const auto& config_sign = walker.ConfigSign();
  for (auto t = 0; t < walker.TimeSize(); ++t) {
    // the factor 1/2 comes from two degenerate spin states ( spin averaged,
    // which is model dependent ) note: gt0 will automatically degenerate to g00
    // if t = 0, it should be safe to replace Greentt with Greent0
    const GreensFunc& gt0 =
        (t == 0) ? 0.5 * (walker.GreenttUp(walker.TimeSize() - 1) +
                          walker.GreenttDn(walker.TimeSize() - 1))
                 : 0.5 * (walker.Greent0Up(t - 1) + walker.Greent0Dn(t - 1));
    density_of_states.tmp_value()(t) +=
        config_sign * gt0.trace() / lattice.SpaceSize();
  }
  ++density_of_states;
}

// The superfluid stiffness rho_s, also known as helicity modules, is defined as
//     rho_s = ( Gamma_L - Gamma_T ) / 4
// where Gamma_L and Gamma_T are longitudinal and horizontal current-current
// (Jx-Jx) correlations in the static (omega = 0) and long wave limit. The
// current-current (Jx-Jx) correlation function Gamma_xx(r,t) in real space is
// defined as
//     Gamma_xx(r,t) = < jx(r,t) * jx(0,0) >
// with the current operator jx(r,t) = i t \sum sigma ( c^+(r+x,t) * c(r,t) -
// c^+(r,t) * c(r+x,t) )(sigma) The superfluid stiffness is useful in locating
// the KT transition temperature of 2d superconducting phase transition. see
// more information in 10.1103/PhysRevB.69.184501
void Methods::measure_superfluid_stiffness(ScalarObs& superfluid_stiffness,
                                           const MeasureHandler& meas_handler,
                                           const Walker& walker,
                                           const ModelBase& model,
                                           const LatticeBase& lattice) {
  DQMC_ASSERT(dynamic_cast<const Lattice::Square*>(&lattice) != nullptr);
  DQMC_ASSERT(lattice.SideLength() % 2 == 0);

  const int space_size = lattice.SpaceSize();
  const int time_size = walker.TimeSize();
  const double config_sign = walker.ConfigSign();
  const double hopping_t2 = model.HoppingT() * model.HoppingT();
  const double final_prefactor =
      0.25 * hopping_t2 / (static_cast<double>(space_size) * space_size);

  std::vector<double> uncorrelated_i_vals(space_size);
  const GreensFunc& g00up = walker.GreenttUp(time_size - 1);
  const GreensFunc& g00dn = walker.GreenttDn(time_size - 1);

  for (auto i = 0; i < space_size; ++i) {
    const auto ipx = lattice.NearestNeighbour(i, 0);
    uncorrelated_i_vals[i] =
        g00up(i, ipx) - g00up(ipx, i) + g00dn(i, ipx) - g00dn(ipx, i);
  }

  double total_rho_s = 0.0;

  for (auto t = 0; t < time_size; ++t) {
    const int tau = (t == 0) ? time_size - 1 : t - 1;

    const GreensFunc& gttup = (t == 0) ? g00up : walker.GreenttUp(tau);
    const GreensFunc& gttdn = (t == 0) ? g00dn : walker.GreenttDn(tau);
    const GreensFunc& gt0up = walker.Greent0Up(tau);
    const GreensFunc& gt0dn = walker.Greent0Dn(tau);
    const GreensFunc& g0tup = walker.Green0tUp(tau);
    const GreensFunc& g0tdn = walker.Green0tDn(tau);

    std::vector<double> uncorrelated_j_vals(space_size);
    for (auto j = 0; j < space_size; ++j) {
      const auto jpx = lattice.NearestNeighbour(j, 0);
      uncorrelated_j_vals[j] =
          gttup(j, jpx) - gttup(jpx, j) + gttdn(j, jpx) - gttdn(jpx, j);
    }

    double t_slice_sum = 0.0;
    for (auto i = 0; i < space_size; ++i) {
      const auto ipx = lattice.NearestNeighbour(i, 0);
      for (auto j = 0; j < space_size; ++j) {
        const auto jpx = lattice.NearestNeighbour(j, 0);

        const auto rx = lattice.Index2Site(lattice.Displacement(i, j), 0);
        const auto ry = lattice.Index2Site(lattice.Displacement(i, j), 1);
        const auto fourier_factor =
            lattice.FourierFactor(rx, 1) - lattice.FourierFactor(ry, 1);

        const double correlated_part =
            g0tup(ipx, jpx) * gt0up(j, i) + g0tdn(ipx, jpx) * gt0dn(j, i) -
            g0tup(i, jpx) * gt0up(j, ipx) - g0tdn(i, jpx) * gt0dn(j, ipx) -
            g0tup(ipx, j) * gt0up(jpx, i) - g0tdn(ipx, j) * gt0dn(jpx, i) +
            g0tup(i, j) * gt0up(jpx, ipx) + g0tdn(i, j) * gt0dn(jpx, ipx);

        t_slice_sum +=
            fourier_factor * (-uncorrelated_j_vals[j] * uncorrelated_i_vals[i] -
                              correlated_part);
      }
    }
    total_rho_s += t_slice_sum;
  }

  superfluid_stiffness.tmp_value() +=
      final_prefactor * config_sign * total_rho_s;
  ++superfluid_stiffness;
}

// transverse relaxation time 1/T1, which is proportional to the (local) dynamic
// spin susceptibility
//
//      1/T1 = 1/N \sum q < Sz(q,t) Sz(q,0) > = 1/N \sim i < Sz(i,t) Sz(i,0) >
//
void Methods::measure_dynamic_spin_susceptibility(
    VectorObs& dynamic_spin_susceptibility, const MeasureHandler& meas_handler,
    const Walker& walker, const ModelBase& model, const LatticeBase& lattice) {
  const int space_size = lattice.SpaceSize();
  const int time_size = walker.TimeSize();
  const double config_sign = walker.ConfigSign();
  const double prefactor = 0.25 * config_sign / space_size;

  const GreensFunc& g00up = walker.GreenttUp(time_size - 1);
  const GreensFunc& g00dn = walker.GreenttDn(time_size - 1);

  for (auto t = 0; t < time_size; ++t) {
    const int tau = (t == 0) ? time_size - 1 : t - 1;

    const GreensFunc& gttup = walker.GreenttUp(tau);
    const GreensFunc& gttdn = walker.GreenttDn(tau);
    const GreensFunc& gt0up = walker.Greent0Up(tau);
    const GreensFunc& gt0dn = walker.Greent0Dn(tau);
    const GreensFunc& g0tup = walker.Green0tUp(tau);
    const GreensFunc& g0tdn = walker.Green0tDn(tau);

    double current_t_sum = 0.0;

    for (auto i = 0; i < space_size; ++i) {
      // g_c(i, i) = (Identity - g.transpose())(i, i) = 1.0 - g(i, i)
      const double gcttup_ii = 1.0 - gttup(i, i);
      const double gcttdn_ii = 1.0 - gttdn(i, i);
      const double gc00up_ii = 1.0 - g00up(i, i);
      const double gc00dn_ii = 1.0 - g00dn(i, i);

      const double up_contribution =
          gcttup_ii * gc00up_ii - g0tup(i, i) * gt0up(i, i);
      const double dn_contribution =
          gcttdn_ii * gc00dn_ii - g0tdn(i, i) * gt0dn(i, i);
      const double mixed_contribution =
          gcttup_ii * gc00dn_ii + gc00up_ii * gcttdn_ii;

      current_t_sum += (up_contribution + dn_contribution - mixed_contribution);
    }
    dynamic_spin_susceptibility.tmp_value()(t) += prefactor * current_t_sum;
  }

  ++dynamic_spin_susceptibility;
}

}  // namespace Measure
