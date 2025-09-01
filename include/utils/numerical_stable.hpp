#pragma once

/**
 *  This head file defines the interface Utils::NumericalStable, which contains subroutines to help
 *  compute equal-time and time-displaced (dynamical) Greens function in a stable manner.
 */

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/QR>

#include "linear_algebra.hpp"
#include "svd_stack.h"
#include "utils/eigen_malloc_guard.h"
#include "utils/temporary_pool.h"

namespace Utils {
namespace NumericalStable {

using namespace LinearAlgebra;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

/*
 *  Subroutine to return the maximum difference of two matrices with the same
 *  size.
 *  Input: umat, vmat
 *  Output: the maximum difference -> error
 */
inline void matrix_compare_error(const Matrix& umat, const Matrix& vmat, double& error) {
  DQMC_ASSERT(umat.rows() == vmat.rows());
  DQMC_ASSERT(umat.cols() == vmat.cols());
  DQMC_ASSERT(umat.rows() == umat.cols());
  error = (umat - vmat).cwiseAbs().maxCoeff();
}

/*
 *  Subroutine to perform the decomposition of a vector, dvec = dmax * dmin,
 *  to ensure all elements that greater than one are stored in dmax,
 *  and all elements that less than one are stored in dmin.
 *  Input: dvec
 *  Output: dmax, dmin
 */
inline void div_dvec_max_min(const Vector& dvec, Vector& dmax, Vector& dmin) {
  DQMC_ASSERT(dvec.size() == dmax.size());
  DQMC_ASSERT(dvec.size() == dmin.size());
  DQMC_ASSERT((dvec.array() >= 0).all());
  dmax = dvec.cwiseMax(Vector::Ones(dvec.size()));
  dmin = dvec.cwiseMin(Vector::Ones(dvec.size()));
}

/*
 *  Subroutine to perform dense matrix * (diagonal matrix)^-1 * dense matrix.
 *  Input: vmat, dvec, umat
 *  Output: zmat
 *  Workspace: temp_mat
 */
inline void mult_v_invd_u(const Matrix& vmat, const Vector& dvec, const Matrix& umat, Matrix& zmat,
                          TemporaryPool& pool) {
  DQMC_ASSERT(vmat.cols() == umat.cols());
  DQMC_ASSERT(vmat.cols() == zmat.cols());
  DQMC_ASSERT(vmat.rows() == umat.rows());
  DQMC_ASSERT(vmat.rows() == zmat.rows());
  DQMC_ASSERT(vmat.rows() == vmat.cols());
  DQMC_ASSERT(vmat.cols() == dvec.size());

  // temp_mat = vmat * D^-1
  auto temp_mat = pool.acquire_matrix(vmat.rows(), vmat.cols());
  temp_mat->noalias() = vmat * dvec.asDiagonal().inverse();
  // zmat = temp_mat * umat
  zmat.noalias() = *temp_mat * umat;
}

/*
 *  Subroutine to perform dense matrix * diagonal matrix * dense matrix.
 *  Input: vmat, dvec, umat
 *  Output: zmat
 *  Workspace: temp_mat
 */
inline void mult_v_d_u(const Matrix& vmat, const Vector& dvec, const Matrix& umat, Matrix& zmat,
                       TemporaryPool& pool) {
  DQMC_ASSERT(vmat.cols() == umat.cols());
  DQMC_ASSERT(vmat.cols() == zmat.cols());
  DQMC_ASSERT(vmat.rows() == umat.rows());
  DQMC_ASSERT(vmat.rows() == zmat.rows());
  DQMC_ASSERT(vmat.rows() == vmat.cols());
  DQMC_ASSERT(vmat.cols() == dvec.size());

  // temp_mat = vmat * D
  auto temp_mat = pool.acquire_matrix(vmat.rows(), vmat.cols());
  temp_mat->noalias() = vmat * dvec.asDiagonal();
  // zmat = temp_mat * umat
  zmat.noalias() = *temp_mat * umat;
}

/*
 * This function applies the following logic element-wise:
 * - If S(i) > 1: Sbi(i) = 1.0 / S(i), Ss(i) = 1.0
 * - If S(i) <= 1: Sbi(i) = 1.0, Ss(i) = S(i)
 *  Input: S
 *  Output: Sbi, Ss
 */
inline void computeSbiSs(const Eigen::VectorXd& S, Eigen::VectorXd& Sbi, Eigen::VectorXd& Ss) {
  DQMC_ASSERT((S.array() >= 0).all());
  DQMC_ASSERT(Sbi.size() == S.size());
  DQMC_ASSERT(Ss.size() == S.size());
  Ss = S.array().min(1.0);
  Sbi = 1.0 / S.array().max(1.0);
}

/*
 *  Applies element-wise scaling to matrices Atmp and Btmp based on
 *  dlmax, drmax, dlmin, and drmin vectors.
 */
inline void scale_Atmp_Btmp_dl_dr(Matrix& Atmp, Matrix& Btmp, const Vector& dlmax,
                                  const Vector& drmax, const Vector& dlmin, const Vector& drmin) {
  DQMC_ASSERT(Atmp.rows() == dlmax.size());
  DQMC_ASSERT(Atmp.cols() == drmax.size());
  DQMC_ASSERT(Btmp.rows() == dlmin.size());
  DQMC_ASSERT(Btmp.cols() == drmin.size());
  DQMC_ASSERT(Atmp.rows() == Btmp.rows());
  DQMC_ASSERT(Atmp.cols() == Btmp.cols());

  Atmp.array().colwise() /= dlmax.array();
  Atmp.array().rowwise() /= drmax.array().transpose();

  Btmp.array().colwise() *= dlmin.array();
  Btmp.array().rowwise() *= drmin.array().transpose();
}

/*
 *  Applies element-wise scaling to matrices Xtmp and Ytmp with swapped vector
 *  roles.
 */
inline void scale_Xtmp_Ytmp_dl_dr(Matrix& Xtmp, Matrix& Ytmp, const Vector& drmax,
                                  const Vector& dlmax, const Vector& drmin, const Vector& dlmin) {
  DQMC_ASSERT(Xtmp.rows() == drmax.size());
  DQMC_ASSERT(Xtmp.cols() == dlmax.size());
  DQMC_ASSERT(Ytmp.rows() == drmin.size());
  DQMC_ASSERT(Ytmp.cols() == dlmin.size());
  DQMC_ASSERT(Xtmp.rows() == Ytmp.rows());
  DQMC_ASSERT(Xtmp.cols() == Ytmp.cols());

  Xtmp.array().colwise() /= drmax.array();
  Xtmp.array().rowwise() /= dlmax.array().transpose();

  Ytmp.array().colwise() *= drmin.array();
  Ytmp.array().rowwise() *= dlmin.array().transpose();
}

/*
 *  return (1 + USV^T)^-1, with method of QR decomposition
 *  to obtain equal-time Green's functions G(t,t).
 */
inline void compute_greens_00_bb(const Matrix& U, const Vector& S, const Matrix& V, Matrix& gtt,
                                 TemporaryPool& pool) {
  const int ndim = S.size();
  auto Sbi = pool.acquire_vector(ndim);
  auto Ss = pool.acquire_vector(ndim);
  computeSbiSs(S, *Sbi, *Ss);

  auto H = pool.acquire_matrix(ndim, ndim);
  auto RHS = pool.acquire_matrix(ndim, ndim);

  RHS->noalias() = Sbi->asDiagonal() * U.transpose();
  H->noalias() = *RHS;
  H->noalias() += Ss->asDiagonal() * V.transpose();

  auto qr_solver = pool.acquire_qr_solver();

  {
    EigenMallocGuard<true> alloc_guard;
    qr_solver->compute(*H);
    gtt.noalias() = qr_solver->solve(*RHS);
  }
}

/*
 *  return (1 + USV^T)^-1 * USV^T, with method of QR decomposition
 *  to obtain time-displaced Green's functions G(beta, 0).
 */
inline void compute_greens_b0(const Matrix& U, const Vector& S, const Matrix& V, Matrix& gt0,
                              TemporaryPool& pool) {
  const int ndim = S.size();
  auto Sbi = pool.acquire_vector(ndim);
  auto Ss = pool.acquire_vector(ndim);
  computeSbiSs(S, *Sbi, *Ss);

  auto H = pool.acquire_matrix(ndim, ndim);
  auto RHS = pool.acquire_matrix(ndim, ndim);

  RHS->noalias() = Ss->asDiagonal() * V.transpose();
  H->noalias() = Sbi->asDiagonal() * U.transpose();
  H->noalias() += *RHS;

  auto qr_solver = pool.acquire_qr_solver();

  {
    EigenMallocGuard<true> alloc_guard;
    qr_solver->compute(*H);
    gt0.noalias() = qr_solver->solve(*RHS);
  }
}

/*
 *  Helper for common calculations in both compute_dynamic_greens and compute_equaltime_greens
 */
inline void compute_greens_function_common_part(const SVD_stack& left, const SVD_stack& right,
                                                Matrix& Atmp, Vector& dlmax, Vector& dlmin,
                                                Vector& drmax, Vector& drmin, TemporaryPool& pool) {
  const int ndim = left.dim();
  const Matrix& ul = left.U();
  const Vector& dl = left.S();
  const Matrix& vl = left.V();
  const Matrix& ur = right.U();
  const Vector& dr = right.S();
  const Matrix& vr = right.V();

  div_dvec_max_min(dl, dlmax, dlmin);
  div_dvec_max_min(dr, drmax, drmin);

  auto Btmp = pool.acquire_matrix(ndim, ndim);
  Atmp.noalias() = ul.transpose() * ur;
  Btmp->noalias() = vl.transpose() * vr;

  scale_Atmp_Btmp_dl_dr(Atmp, *Btmp, dlmax, drmax, dlmin, drmin);

  auto tmp = pool.acquire_matrix(ndim, ndim);
  tmp->noalias() = Atmp + *Btmp;

  auto B_for_solve = pool.acquire_matrix(ndim, ndim);
  B_for_solve->noalias() = ur * drmax.asDiagonal().inverse();

  auto qr_solver = pool.acquire_qr_solver();
  solve_X_times_A_eq_B(Atmp, *tmp, *B_for_solve, *qr_solver);
}

/*
 *  return (1 + left * right^T)^-1 in a stable manner.
 */
inline void compute_equaltime_greens(const SVD_stack& left, const SVD_stack& right, Matrix& gtt,
                                     TemporaryPool& pool) {
  DQMC_ASSERT(left.dim() == right.dim());
  const int ndim = left.dim();

  if (left.empty()) {
    compute_greens_00_bb(right.V(), right.S(), right.U(), gtt, pool);
    return;
  }
  if (right.empty()) {
    compute_greens_00_bb(left.U(), left.S(), left.V(), gtt, pool);
    return;
  }

  auto Atmp = pool.acquire_matrix(ndim, ndim);
  auto dlmax = pool.acquire_vector(ndim);
  auto dlmin = pool.acquire_vector(ndim);
  auto drmax = pool.acquire_vector(ndim);
  auto drmin = pool.acquire_vector(ndim);

  auto left_U_transp = pool.acquire_matrix(ndim, ndim);
  left_U_transp->noalias() = left.U().transpose();

  compute_greens_function_common_part(left, right, *Atmp, *dlmax, *dlmin, *drmax, *drmin, pool);
  mult_v_invd_u(*Atmp, *dlmax, *left_U_transp, gtt, pool);
}

/*
 *  return time-displaced Green's function in a stable manner.
 */
inline void compute_dynamic_greens(const SVD_stack& left, const SVD_stack& right, Matrix& gt0,
                                   Matrix& g0t, TemporaryPool& pool) {
  DQMC_ASSERT(left.dim() == right.dim());
  const int ndim = left.dim();

  if (left.empty()) {
    compute_greens_00_bb(right.V(), right.S(), right.U(), gt0, pool);
    g0t.noalias() = gt0;
    g0t.diagonal().array() -= 1.0;
    return;
  }

  if (right.empty()) {
    compute_greens_b0(left.U(), left.S(), left.V(), gt0, pool);
    compute_greens_00_bb(left.U(), left.S(), left.V(), g0t, pool);
    g0t *= -1.0;
    return;
  }

  auto dlmax = pool.acquire_vector(ndim);
  auto dlmin = pool.acquire_vector(ndim);
  auto drmax = pool.acquire_vector(ndim);
  auto drmin = pool.acquire_vector(ndim);

  // Part 1: compute gt0
  {
    auto Atmp = pool.acquire_matrix(ndim, ndim);
    compute_greens_function_common_part(left, right, *Atmp, *dlmax, *dlmin, *drmax, *drmin, pool);

    auto left_V_transp = pool.acquire_matrix(ndim, ndim);
    left_V_transp->noalias() = left.V().transpose();

    mult_v_d_u(*Atmp, *dlmin, *left_V_transp, gt0, pool);
  }

  // Part 2: compute g0t
  {
    const Matrix& vl = left.V();
    const Matrix& ur = right.U();
    const Matrix& vr = right.V();

    auto Xtmp = pool.acquire_matrix(ndim, ndim);
    Xtmp->noalias() = vr.transpose() * vl;

    auto Ytmp = pool.acquire_matrix(ndim, ndim);
    Ytmp->noalias() = ur.transpose() * left.U();

    scale_Xtmp_Ytmp_dl_dr(*Xtmp, *Ytmp, *drmax, *dlmax, *drmin, *dlmin);

    auto tmp = pool.acquire_matrix(ndim, ndim);
    tmp->noalias() = *Xtmp + *Ytmp;

    auto B_for_solve = pool.acquire_matrix(ndim, ndim);
    B_for_solve->noalias() = (-vl) * dlmax->asDiagonal().inverse();

    auto qr_solver = pool.acquire_qr_solver();
    solve_X_times_A_eq_B(*Xtmp, *tmp, *B_for_solve, *qr_solver);

    auto ur_transp = pool.acquire_matrix(ndim, ndim);
    ur_transp->noalias() = ur.transpose();

    mult_v_d_u(*Xtmp, *drmin, *ur_transp, g0t, pool);
  }
}
}  // namespace NumericalStable
}  // namespace Utils
