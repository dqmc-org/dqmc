#include "measure/measure_methods.h"

#include "lattice/lattice_base.h"
#include "lattice/square.h"
#include "measure/measure_handler.h"
#include "model/model_base.h"
#include "walker.h"

namespace Measure {

// some aliases
using GreensFunc = Eigen::MatrixXd;
using MatrixType = Eigen::MatrixXd;
using VectorType = Eigen::VectorXd;

// -----------------------------  Method routines for equal-time measurements
// -------------------------------

// The sign of the bosonic field configurations for equal-time measurements,
// useful for reweighting
void Methods::measure_equaltime_config_sign(Observable::Scalar& equaltime_sign,
                                            const MeasureContext& ctx) {
  equaltime_sign.accumulate(ctx.walker.vec_config_sign().sum(), ctx.walker.time_size());
}

// Filling number defined as \sum i ( n_up + n_dn )(i)
// which represents the total number of electrons
void Methods::measure_filling_number(Observable::Scalar& filling_number,
                                     const MeasureContext& ctx) {
  const int time_size = ctx.walker.time_size();
  const int space_size = ctx.lattice.space_size();

  for (auto t = 0; t < time_size; ++t) {
    const auto& g_up = ctx.walker.green_tt_up(t);
    const auto& g_dn = ctx.walker.green_tt_down(t);
    const double config_sign = ctx.walker.config_sign(t);

    double combined_trace = g_up.trace() + g_dn.trace();
    const double avg_density = combined_trace / space_size;

    filling_number.accumulate(config_sign * (2.0 - avg_density));
  }
}

// Double occupation defined as \sum i ( n_up * n_dn )(i)
// which quantizes the possibility that two electrons with opposite spin occupy
// the same site
void Methods::measure_double_occupancy(Observable::Scalar& double_occupancy,
                                       const MeasureContext& ctx) {
  const int time_size = ctx.walker.time_size();
  const int space_size = ctx.lattice.space_size();

  for (auto t = 0; t < time_size; ++t) {
    const GreensFunc& gu = ctx.walker.green_tt_up(t);
    const GreensFunc& gd = ctx.walker.green_tt_down(t);
    const double config_sign = ctx.walker.config_sign(t);

    double sum = 0.0;
    for (int i = 0; i < space_size; ++i) {
      sum += (1.0 - gu(i, i)) * (1.0 - gd(i, i));
    }

    double_occupancy.accumulate(config_sign * sum / space_size);
  }
}

// Kinetic energy defined as -t \sum <ij> ( c^+_j c_i + h.c. )
// which measures the hoppings between two sites, e.g. hoppings between nearest
// neighbours
void Methods::measure_kinetic_energy(Observable::Scalar& kinetic_energy,
                                     const MeasureContext& ctx) {
  const int time_size = ctx.walker.time_size();
  const int space_size = ctx.lattice.space_size();
  const double t_hop = ctx.model.HoppingT();

  for (auto t = 0; t < time_size; ++t) {
    const GreensFunc& g_up = ctx.walker.green_tt_up(t);
    const GreensFunc& g_dn = ctx.walker.green_tt_down(t);
    const double config_sign = ctx.walker.config_sign(t);

    double sum = 0.0;
    for (auto i = 0; i < space_size; ++i) {
      for (const auto j : ctx.lattice.get_neighbors(i)) {
        sum += g_up(i, j) + g_dn(i, j);
      }
    }

    kinetic_energy.accumulate(config_sign * t_hop * sum / space_size);
  }
}

// Total energy: E = Kinetic + U * DoubleOccupancy - mu * Filling
void Methods::measure_total_energy(Observable::Scalar& total_energy, const MeasureContext& ctx) {
  const int time_size = ctx.walker.time_size();
  const int space_size = ctx.lattice.space_size();

  const double t_hop = ctx.model.HoppingT();
  const double U = ctx.model.OnSiteU();
  const double mu = ctx.model.ChemicalPotential();

  for (auto tt = 0; tt < time_size; ++tt) {
    const auto& g_up = ctx.walker.green_tt_up(tt);
    const auto& g_dn = ctx.walker.green_tt_down(tt);
    const double config_sign = ctx.walker.config_sign(tt);

    double sum_hop = 0.0;
    for (int i = 0; i < space_size; ++i) {
      for (const auto j : ctx.lattice.get_neighbors(i)) {
        sum_hop += g_up(i, j) + g_dn(i, j);
      }
    }
    double kinetic = config_sign * t_hop * sum_hop / space_size;

    double sum_double_t = 0.0;
    for (int i = 0; i < space_size; ++i) {
      sum_double_t += (1.0 - g_up(i, i)) * (1.0 - g_dn(i, i));
    }
    double double_occ = config_sign * sum_double_t / space_size;
    double potential = U * double_occ;

    double combined_trace = g_up.trace() + g_dn.trace();
    double density_per_site = 2.0 - (combined_trace / space_size);
    double mu_term = config_sign * (-mu) * density_per_site;

    total_energy.accumulate(kinetic + potential + mu_term);
  }
}

// In general, spin correlations is defined as C(i,t) = < (n_up - n_dn)(i,t) *
// (n_up - n_dn)(0,0) > which measure the correlations of spins between two
// space-time points. the local correlations are the limit of i = 0 and t = 0.
void Methods::measure_local_spin_corr(Observable::Scalar& local_spin_corr,
                                      const MeasureContext& ctx) {
  const int time_size = ctx.walker.time_size();
  const int space_size = ctx.lattice.space_size();

  for (auto t = 0; t < time_size; ++t) {
    const GreensFunc& gu = ctx.walker.green_tt_up(t);
    const GreensFunc& gd = ctx.walker.green_tt_down(t);
    const double config_sign = ctx.walker.config_sign(t);

    double accum = 0.0;

    for (auto i = 0; i < space_size; ++i) {
      // Let n_up = gu(i, i) and n_dn = gd(i, i). The local spin correlation
      // <S_z^2> is proportional to <(n_up - n_dn)^2> = <n_up^2 - 2n_up*n_dn +
      // n_dn^2>. For fermions, n^2 = n, so this is <n_up + n_dn - 2*n_up*n_dn>.
      const double n_up = gu(i, i);
      const double n_dn = gd(i, i);
      accum += config_sign * (n_up + n_dn - 2.0 * n_up * n_dn);
    }

    local_spin_corr.accumulate(accum / space_size);
  }
}

// Distribution of electrons in momentum space defined as n(k) = ( n_up + n_dn
// )(k) measured for one specific momentum point todo: scan the momentum space
void Methods::measure_momentum_distribution(Observable::Scalar& momentum_dist,
                                            const MeasureContext& ctx) {
  const int time_size = ctx.walker.time_size();
  const int space_size = ctx.lattice.space_size();
  const int K_vector = ctx.handler.momentum();
  const double norm_factor = 0.5 / static_cast<double>(space_size);

  const auto& fourier_factor = ctx.lattice.fourier_factor();
  const auto& displacement = ctx.lattice.displacement();

  for (auto t = 0; t < time_size; ++t) {
    const GreensFunc& gu = ctx.walker.green_tt_up(t);
    const GreensFunc& gd = ctx.walker.green_tt_down(t);
    const double config_sign = ctx.walker.config_sign(t);

    double sum = 0.0;

    for (auto j = 0; j < space_size; ++j) {
      for (auto i = 0; i < space_size; ++i) {
        sum += (gu(j, i) + gd(j, i)) * fourier_factor(displacement(i, j), K_vector);
      }
    }

    momentum_dist.accumulate(config_sign * (1 - norm_factor * sum));
  }
}

// Structure factor of spin density wave (SDW) defined as
// 1/N \sum ij ( exp( -i Q*(ri-rj) ) * (n_up - n_dn)(j) * (n_up - n_dn)(i) )
// where Q is the wave momentum of sdw.
void Methods::measure_spin_density_structure_factor(Observable::Scalar& sdw_factor,
                                                    const MeasureContext& ctx) {
  const int space_size = ctx.lattice.space_size();
  const int K_vector = ctx.handler.momentum();

  const auto& fourier_factor = ctx.lattice.fourier_factor();
  const auto& displacement = ctx.lattice.displacement();

  auto tmp1 = ctx.pool.acquire_matrix(space_size, space_size);
  auto tmp2 = ctx.pool.acquire_matrix(space_size, space_size);

  MatrixType& guc = *tmp1;
  MatrixType& gdc = *tmp2;

  for (auto t = 0; t < ctx.walker.time_size(); ++t) {
    const GreensFunc& gu = ctx.walker.green_tt_up(t);
    const GreensFunc& gd = ctx.walker.green_tt_down(t);
    const double config_sign = ctx.walker.config_sign(t);

    // We need to make this dance to avoid allocations
    guc.setIdentity();
    guc -= gu.transpose();
    gdc.setIdentity();
    gdc -= gd.transpose();

    double sum = 0.0;
    for (auto i = 0; i < space_size; ++i) {
      const double guc_ii = guc(i, i);
      const double gdc_ii = gdc(i, i);

      for (auto j = 0; j < space_size; ++j) {
        sum += fourier_factor(displacement(i, j), K_vector) *
               (+guc_ii * guc(j, j) + guc(i, j) * gu(i, j) + gdc_ii * gdc(j, j) +
                gdc(i, j) * gd(i, j) - gdc_ii * guc(j, j) - guc_ii * gdc(j, j));
      }
    }

    sdw_factor.accumulate(config_sign * sum / space_size / space_size);
  }
}
// Structure factor of charge density wave (CDW) defined as
// 1/N \sum ij ( exp( -i Q*(ri-rj) ) * (n_up + n_dn)(j) * (n_up + n_dn)(i) )
// where Q is the wave momentum of cdw.
void Methods::measure_charge_density_structure_factor(Observable::Scalar& cdw_factor,
                                                      const MeasureContext& ctx) {
  const int space_size = ctx.lattice.space_size();
  const int time_size = ctx.walker.time_size();
  const int K_vector = ctx.handler.momentum();

  const auto& fourier_factor = ctx.lattice.fourier_factor();
  const auto& displacement = ctx.lattice.displacement();

  auto tmp1 = ctx.pool.acquire_vector(space_size);
  auto tmp2 = ctx.pool.acquire_vector(space_size);

  Eigen::VectorXd& n_up = *tmp1;
  Eigen::VectorXd& n_dn = *tmp2;

  for (auto t = 0; t < time_size; ++t) {
    const GreensFunc& gu = ctx.walker.green_tt_up(t);
    const GreensFunc& gd = ctx.walker.green_tt_down(t);
    const double config_sign = ctx.walker.config_sign(t);

    for (int i = 0; i < space_size; ++i) {
      n_up(i) = 1.0 - gu(i, i);
      n_dn(i) = 1.0 - gd(i, i);
    }

    double sum = 0.0;
    for (auto i = 0; i < space_size; ++i) {
      const double ni = n_up(i) + n_dn(i);

      for (auto j = 0; j < space_size; ++j) {
        const double nj = n_up(j) + n_dn(j);
        const double density_term = ni * nj;

        const double guc_ij = (i == j) - gu(j, i);
        const double gdc_ij = (i == j) - gd(j, i);
        const double hopping_term = guc_ij * gu(i, j) + gdc_ij * gd(i, j);
        const double total_correlator = density_term + hopping_term;

        sum += fourier_factor(displacement(i, j), K_vector) * total_correlator;
      }
    }
    cdw_factor.accumulate(config_sign * sum / space_size / space_size);
  }
}

// The s-wave superconducting pairing is defined as Delta = 1/sqrt(N) \sum i (
// c_up * c_dn )(i) Accordingly, the correlation function Ps reads
//   Ps  =  1/2 ( Delta^+ * Delta + h.c. )
//       =  1/N \sum ij ( (delta_ij - Gup(j,i)) * (delta_ij - Gdn(j,i)) )
// which serves as the Laudau order paramerter, and, with special attention, is
// an extensive quantity. Note the 1/2 prefactor in definition of Ps cancels the
// duplicated countings of ij.
void Methods::measure_s_wave_pairing_corr(Observable::Scalar& s_wave_pairing,
                                          const MeasureContext& ctx) {
  const int space_size = ctx.lattice.space_size();

  auto tmp1 = ctx.pool.acquire_matrix(space_size, space_size);
  auto tmp2 = ctx.pool.acquire_matrix(space_size, space_size);

  MatrixType& guc = *tmp1;
  MatrixType& gdc = *tmp2;

  for (auto t = 0; t < ctx.walker.time_size(); ++t) {
    //  g(i,j) = < c_i * c^+_j > are the greens functions
    // gc(i,j) = < c^+_i * c_j > are isomorphic to the conjugation of greens
    // functions
    const GreensFunc& gu = ctx.walker.green_tt_up(t);
    const GreensFunc& gd = ctx.walker.green_tt_down(t);

    // We need to make this dance to avoid allocations
    guc.setIdentity();
    guc -= gu.transpose();
    gdc.setIdentity();
    gdc -= gd.transpose();

    const double& config_sign = ctx.walker.config_sign(t);

    // loop over site i, j and take averages
    double sum = 0.0;
    for (auto i = 0; i < space_size; ++i) {
      for (auto j = 0; j < space_size; ++j) {
        sum += config_sign * (guc(i, j) * gdc(i, j));
      }
    }
    // entensive quantity
    s_wave_pairing.accumulate(sum / space_size);
  }
}

// ------------------------------  Method routines for dynamic measurements
// ---------------------------------

// The sign of the bosonic field configurations for dynamic measurements
void Methods::measure_dynamic_config_sign(Observable::Scalar& dynamic_sign,
                                          const MeasureContext& ctx) {
  dynamic_sign.accumulate(ctx.walker.config_sign());
}

// Green's functions G(k,t) = < c(k,t) c^+(k,0) > in momentum space
// which are defined as the Fourier transmations of G(i,j) in real space
// G(k,t)  =  1/N \sum ij exp( -i k*(rj-ri) ) * ( c_j(t) * c^+_i(0) )
void Methods::measure_greens_functions(Observable::Matrix& greens_functions,
                                       const MeasureContext& ctx) {
  const int time_size = ctx.walker.time_size();
  const int space_size = ctx.lattice.space_size();
  const int num_momenta = ctx.handler.momentum_list().size();
  const double config_sign = ctx.walker.config_sign();

  // Self-size the tmp_value to match (num_momenta, time_size) dimensions
  if (greens_functions.accumulator().rows() != num_momenta ||
      greens_functions.accumulator().cols() != time_size) {
    EigenMallocGuard<true> alloc_guard;
    greens_functions.accumulator().resize(num_momenta, time_size);
    greens_functions.accumulator().setZero();
  }

  auto tmp = ctx.pool.acquire_matrix(num_momenta, time_size);
  Eigen::MatrixXd& result = *tmp;
  result.setZero();

  for (auto t = 0; t < time_size; ++t) {
    const int tau = (t == 0) ? time_size - 1 : t - 1;

    const GreensFunc& gup = (t == 0) ? ctx.walker.green_tt_up(tau) : ctx.walker.green_t0_up(tau);
    const GreensFunc& gdn =
        (t == 0) ? ctx.walker.green_tt_down(tau) : ctx.walker.green_t0_down(tau);

    for (auto k = 0; k < num_momenta; ++k) {
      const auto& K_vector = ctx.handler.momentum_list(k);

      double current_k_t_sum = 0.0;

      for (auto i = 0; i < space_size; ++i) {
        for (auto j = 0; j < space_size; ++j) {
          const double gt0_ji = 0.5 * (gup(j, i) + gdn(j, i));

          const auto fourier_factor =
              ctx.lattice.fourier_factor(ctx.lattice.displacement(i, j), K_vector);

          current_k_t_sum += gt0_ji * fourier_factor;
        }
      }
      result(k, t) += config_sign * current_k_t_sum / space_size;
    }
  }
  greens_functions.accumulate(result);
}

// Density of states D(t) defined as 1/N \sum i ( c(i,t) * c^+(i,0) )
// whose fourier transformations are exactly the usual density of states
// D(omega).
void Methods::measure_density_of_states(Observable::Vector& density_of_states,
                                        const MeasureContext& ctx) {
  const int time_size = ctx.walker.time_size();
  const int space_size = ctx.lattice.space_size();
  const auto& config_sign = ctx.walker.config_sign();

  // Self-size the tmp_value to match time_size (tau dimension)
  if (density_of_states.accumulator().size() != time_size) {
    EigenMallocGuard<true> alloc_guard;
    density_of_states.accumulator().resize(time_size);
    density_of_states.accumulator().setZero();
  }

  auto tmp = ctx.pool.acquire_vector(time_size);
  Eigen::VectorXd& result = *tmp;
  result.setZero();

  for (auto t = 0; t < time_size; ++t) {
    const int tau = (t == 0) ? time_size - 1 : t - 1;

    const GreensFunc& gup = (t == 0) ? ctx.walker.green_tt_up(tau) : ctx.walker.green_t0_up(tau);
    const GreensFunc& gdn =
        (t == 0) ? ctx.walker.green_tt_down(tau) : ctx.walker.green_t0_down(tau);

    const double gt0_trace = 0.5 * (gup.trace() + gdn.trace());

    result(t) += config_sign * gt0_trace / space_size;
  }
  density_of_states.accumulate(result);
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
void Methods::measure_superfluid_stiffness(Observable::Scalar& superfluid_stiffness,
                                           const MeasureContext& ctx) {
  DQMC_ASSERT(dynamic_cast<const Lattice::Square*>(&ctx.lattice) != nullptr);
  DQMC_ASSERT(ctx.lattice.side_length() % 2 == 0);

  const int space_size = ctx.lattice.space_size();
  const int time_size = ctx.walker.time_size();
  const double config_sign = ctx.walker.config_sign();
  const double t_hop = ctx.model.HoppingT();

  auto tmp1 = ctx.pool.acquire_vector(space_size);
  auto tmp2 = ctx.pool.acquire_vector(space_size);

  Eigen::VectorXd& uncorrelated_i_vals = *tmp1;
  Eigen::VectorXd& uncorrelated_j_vals = *tmp2;

  const GreensFunc& g00up = ctx.walker.green_tt_up(time_size - 1);
  const GreensFunc& g00dn = ctx.walker.green_tt_down(time_size - 1);

  for (auto i = 0; i < space_size; ++i) {
    const auto ipx = ctx.lattice.nearest_neighbor(i, 0);
    uncorrelated_i_vals(i) = g00up(i, ipx) - g00up(ipx, i) + g00dn(i, ipx) - g00dn(ipx, i);
  }

  double result = 0.0;

  for (auto t = 0; t < time_size; ++t) {
    const int tau = (t == 0) ? time_size - 1 : t - 1;

    const GreensFunc& gttup = (t == 0) ? g00up : ctx.walker.green_tt_up(tau);
    const GreensFunc& gttdn = (t == 0) ? g00dn : ctx.walker.green_tt_down(tau);
    const GreensFunc& gt0up = ctx.walker.green_t0_up(tau);
    const GreensFunc& gt0dn = ctx.walker.green_t0_down(tau);
    const GreensFunc& g0tup = ctx.walker.green_0t_up(tau);
    const GreensFunc& g0tdn = ctx.walker.green_0t_down(tau);

    for (auto j = 0; j < space_size; ++j) {
      const auto jpx = ctx.lattice.nearest_neighbor(j, 0);
      uncorrelated_j_vals(j) = gttup(j, jpx) - gttup(jpx, j) + gttdn(j, jpx) - gttdn(jpx, j);
    }

    double time_slice_sum = 0.0;
    for (auto i = 0; i < space_size; ++i) {
      const auto ipx = ctx.lattice.nearest_neighbor(i, 0);
      for (auto j = 0; j < space_size; ++j) {
        const auto jpx = ctx.lattice.nearest_neighbor(j, 0);

        const auto displacement = ctx.lattice.displacement(i, j);
        const auto rx = ctx.lattice.index_to_site(displacement, 0);
        const auto ry = ctx.lattice.index_to_site(displacement, 1);
        const auto fourier_factor =
            ctx.lattice.fourier_factor(rx, 1) - ctx.lattice.fourier_factor(ry, 1);

        const double up_contribution =
            g0tup(ipx, jpx) * gt0up(j, i) - g0tup(i, jpx) * gt0up(j, ipx) -
            g0tup(ipx, j) * gt0up(jpx, i) + g0tup(i, j) * gt0up(jpx, ipx);

        const double down_contribution =
            g0tdn(ipx, jpx) * gt0dn(j, i) - g0tdn(i, jpx) * gt0dn(j, ipx) -
            g0tdn(ipx, j) * gt0dn(jpx, i) + g0tdn(i, j) * gt0dn(jpx, ipx);

        const double correlated_part = up_contribution + down_contribution;

        time_slice_sum +=
            fourier_factor * (-uncorrelated_j_vals(j) * uncorrelated_i_vals(i) - correlated_part);
      }
    }
    result += time_slice_sum;
  }

  superfluid_stiffness.accumulate(0.25 * t_hop * t_hop * config_sign * result / space_size /
                                  space_size);
}

// transverse relaxation time 1/T1, which is proportional to the (local) dynamic
// spin susceptibility
//
//      1/T1 = 1/N \sum q < Sz(q,t) Sz(q,0) > = 1/N \sim i < Sz(i,t) Sz(i,0) >
//
void Methods::measure_dynamic_spin_susceptibility(Observable::Vector& dynamic_spin_susceptibility,
                                                  const MeasureContext& ctx) {
  const int space_size = ctx.lattice.space_size();
  const int time_size = ctx.walker.time_size();
  const double config_sign = ctx.walker.config_sign();

  // Self-size the tmp_value to match time_size (tau dimension)
  if (dynamic_spin_susceptibility.accumulator().size() != time_size) {
    EigenMallocGuard<true> alloc_guard;
    dynamic_spin_susceptibility.accumulator().resize(time_size);
    dynamic_spin_susceptibility.accumulator().setZero();
  }

  const GreensFunc& g00up = ctx.walker.green_tt_up(time_size - 1);
  const GreensFunc& g00dn = ctx.walker.green_tt_down(time_size - 1);

  auto tmp = ctx.pool.acquire_vector(time_size);
  Eigen::VectorXd& result = *tmp;
  result.setZero();

  for (auto t = 0; t < time_size; ++t) {
    const int tau = (t == 0) ? time_size - 1 : t - 1;

    const GreensFunc& gttup = ctx.walker.green_tt_up(tau);
    const GreensFunc& gttdn = ctx.walker.green_tt_down(tau);
    const GreensFunc& gt0up = ctx.walker.green_t0_up(tau);
    const GreensFunc& gt0dn = ctx.walker.green_t0_down(tau);
    const GreensFunc& g0tup = ctx.walker.green_0t_up(tau);
    const GreensFunc& g0tdn = ctx.walker.green_0t_down(tau);

    double current_time_sum = 0.0;

    for (auto i = 0; i < space_size; ++i) {
      // g_c(i, i) = (Identity - g.transpose())(i, i) = 1.0 - g(i, i)
      const double gcttup_ii = 1.0 - gttup(i, i);
      const double gcttdn_ii = 1.0 - gttdn(i, i);
      const double gc00up_ii = 1.0 - g00up(i, i);
      const double gc00dn_ii = 1.0 - g00dn(i, i);

      const double up_contribution = gcttup_ii * gc00up_ii - g0tup(i, i) * gt0up(i, i);
      const double dn_contribution = gcttdn_ii * gc00dn_ii - g0tdn(i, i) * gt0dn(i, i);
      const double mixed_contribution = gcttup_ii * gc00dn_ii + gc00up_ii * gcttdn_ii;

      current_time_sum += (up_contribution + dn_contribution - mixed_contribution);
    }
    result(t) += 0.25 * config_sign * current_time_sum / space_size;
  }

  dynamic_spin_susceptibility.accumulate(result);
}

// Pair–pair correlation function at momenta Q (vector over momentum_list):
//   P(Q) = < d_Q^† d_Q >
//        = (1/N) sum_{i,j} exp(i Q · (r_i - r_j))
//          * < c†_{i,down} c†_{i,up} c_{j,up} c_{j,down} >
// Evaluated via Wick contractions using equal-time Green's.
void Methods::measure_pair_pair_corr_Q(Observable::Vector& pair_corr_Q, const MeasureContext& ctx) {
  const int space_size = ctx.lattice.space_size();
  const int time_size = ctx.walker.time_size();
  const int num_momenta = ctx.handler.momentum_list().size();

  // Self-size the tmp_value to match number of momentum vectors
  if (pair_corr_Q.accumulator().size() != num_momenta) {
    EigenMallocGuard<true> alloc_guard;
    pair_corr_Q.accumulator().resize(num_momenta);
    pair_corr_Q.accumulator().setZero();
  }

  auto tmp = ctx.pool.acquire_vector(num_momenta);
  Eigen::VectorXd& result = *tmp;
  result.setZero();

  for (int t = 0; t < time_size; ++t) {
    const GreensFunc& gu = ctx.walker.green_tt_up(t);    // G_up(i,j) = < c_i c_j^† >
    const GreensFunc& gd = ctx.walker.green_tt_down(t);  // G_dn(i,j)
    const double config_sign = ctx.walker.config_sign(t);

    for (int k = 0; k < num_momenta; ++k) {
      const auto& Q = ctx.handler.momentum_list(k);  // momentum vector
      double sum_ij = 0.0;

      for (int i = 0; i < space_size; ++i) {
        for (int j = 0; j < space_size; ++j) {
          // < c_i^† c_j > = δ_ij - G(j,i)
          const double gij_up = (i == j ? 1.0 : 0.0) - gu(j, i);
          const double gij_dn = (i == j ? 1.0 : 0.0) - gd(j, i);

          const double fourier = ctx.lattice.fourier_factor(ctx.lattice.displacement(i, j), Q);

          sum_ij += fourier * gij_up * gij_dn;
        }
      }

      result(k) += config_sign * sum_ij / space_size;
    }
  }

  pair_corr_Q.accumulate(result);
}

// Dynamic pair correlator P(Q, tau) = < d_Q(tau) d_Q^†(0) >
// where d_Q = (1/sqrt(N)) sum_i e^{-i Q · r_i} c_{i↑} c_{i↓}.
// Returns a Vector over tau (indexed like other dynamic Vector observables).
void Methods::measure_dynamic_pair_corr(Observable::Vector& dynamic_pair_corr,
                                        const MeasureContext& ctx) {
  const int time_size = ctx.walker.time_size();
  const int space_size = ctx.lattice.space_size();
  const int K_vector = ctx.handler.momentum();  // single momentum index/vector

  const auto& fourier_factor = ctx.lattice.fourier_factor();
  const auto& displacement = ctx.lattice.displacement();

  // Self-size the tmp_value to match time_size (tau dimension)
  if (dynamic_pair_corr.accumulator().size() != time_size) {
    EigenMallocGuard<true> alloc_guard;
    dynamic_pair_corr.accumulator().resize(time_size);
    dynamic_pair_corr.accumulator().setZero();
  }

  auto tmp = ctx.pool.acquire_vector(time_size);
  Eigen::VectorXd& result = *tmp;
  result.setZero();

  for (auto t = 0; t < time_size; ++t) {
    const int tau = (t == 0) ? time_size - 1 : t - 1;

    const GreensFunc& gup = (t == 0) ? ctx.walker.green_tt_up(tau) : ctx.walker.green_t0_up(tau);
    const GreensFunc& gdn =
        (t == 0) ? ctx.walker.green_tt_down(tau) : ctx.walker.green_t0_down(tau);

    const double config_sign = ctx.walker.config_sign();

    double sum_ij = 0.0;
    // double sum over site indices i,j
    for (auto i = 0; i < space_size; ++i) {
      for (auto j = 0; j < space_size; ++j) {
        const double G_up_ij = gup(i, j);
        const double G_dn_ij = gdn(i, j);
        sum_ij += fourier_factor(displacement(i, j), K_vector) * (G_up_ij * G_dn_ij);
      }
    }
    result(t) += config_sign * sum_ij / space_size;
  }

  dynamic_pair_corr.accumulate(result);
}

}  // namespace Measure
