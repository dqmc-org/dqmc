#pragma once

/**
 *  This head file defines the interface Utils::NumericalStable class,
 *  which contains subroutines to help compute equal-time and
 *  time-displaced (dynamical) Greens function in a stable manner.
 */

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/QR>

#include "linear_algebra.hpp"
#include "svd_stack.h"

namespace Utils {

struct GreensWorkspace {
  int ndim = 0;

  Eigen::VectorXd dlmax, dlmin, drmax, drmin;
  Eigen::MatrixXd Atmp, Btmp, Xtmp, Ytmp, tmp, B_for_solve;

  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr_solver;

  GreensWorkspace() = default;
  ~GreensWorkspace() = default;
  GreensWorkspace(const GreensWorkspace&) = delete;
  GreensWorkspace& operator=(const GreensWorkspace&) = delete;
  GreensWorkspace(GreensWorkspace&&) = default;
  GreensWorkspace& operator=(GreensWorkspace&&) = default;

  void resize(int n);
};

// ----------------------------------  Utils::NumericalStable class
// ------------------------------------ including static subroutines for
// numerical stabilizations
class NumericalStable {
 public:
  using Matrix = Eigen::MatrixXd;
  using Vector = Eigen::VectorXd;

  /*
   *  Subroutine to return the maximum difference of two matrices with the same
   * size. Input: umat, vmat Output: the maximum difference -> error
   */
  static void matrix_compare_error(const Matrix& umat, const Matrix& vmat, double& error) {
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
  static void div_dvec_max_min(const Vector& dvec, Vector& dmax, Vector& dmin) {
    DQMC_ASSERT(dvec.size() == dmax.size());
    DQMC_ASSERT(dvec.size() == dmin.size());
    DQMC_ASSERT((dvec.array() >= 0).all());
    dmax = dvec.cwiseMax(Vector::Ones(dvec.size()));
    dmin = dvec.cwiseMin(Vector::Ones(dvec.size()));
  }

  /*
   *  Subroutine to perform dense matrix * (diagonal matrix)^-1 * dense matrix
   *  Input: vmat, dvec, umat
   *  Output: zmat
   */
  static void mult_v_invd_u(const Matrix& vmat, const Vector& dvec, const Matrix& umat,
                            Matrix& zmat) {
    DQMC_ASSERT(vmat.cols() == umat.cols());
    DQMC_ASSERT(vmat.cols() == zmat.cols());
    DQMC_ASSERT(vmat.rows() == umat.rows());
    DQMC_ASSERT(vmat.rows() == zmat.rows());
    DQMC_ASSERT(vmat.rows() == vmat.cols());
    DQMC_ASSERT(vmat.cols() == dvec.size());
    zmat.noalias() = vmat * dvec.asDiagonal().inverse() * umat;
  }

  /*
   *  Subroutine to perform dense matrix * diagonal matrix * dense matrix
   *  Input: vmat, dvec, umat
   *  Output: zmat
   */
  static void mult_v_d_u(const Matrix& vmat, const Vector& dvec, const Matrix& umat, Matrix& zmat) {
    DQMC_ASSERT(vmat.cols() == umat.cols());
    DQMC_ASSERT(vmat.cols() == zmat.cols());
    DQMC_ASSERT(vmat.rows() == umat.rows());
    DQMC_ASSERT(vmat.rows() == zmat.rows());
    DQMC_ASSERT(vmat.rows() == vmat.cols());
    DQMC_ASSERT(vmat.cols() == dvec.size());
    zmat.noalias() = vmat * dvec.asDiagonal() * umat;
  }

  /*
   * This function applies the following logic element-wise:
   * - If S(i) > 1: Sbi(i) = 1.0 / S(i), Ss(i) = 1.0
   * - If S(i) <= 1: Sbi(i) = 1.0, Ss(i) = S(i)
   *  Input: S
   *  Output: Sbi, Ss
   */
  static void computeSbiSs(const Eigen::VectorXd& S, Eigen::VectorXd& Sbi, Eigen::VectorXd& Ss) {
    DQMC_ASSERT((S.array() >= 0).all());
    DQMC_ASSERT(Sbi.size() == S.size());
    DQMC_ASSERT(Ss.size() == S.size());
    Ss = S.array().min(1.0);
    Sbi = 1.0 / S.array().max(1.0);
  }

  /*
   *  Applies element-wise scaling to matrices Atmp and Btmp based on
   *  dlmax, drmax, dlmin, and drmin vectors.
   *
   *  Specifically:
   *  Atmp(i, j) = Atmp(i, j) / (dlmax(i) * drmax(j))
   *  Btmp(i, j) = Btmp(i, j) * (dlmin(i) * drmin(j))
   *
   *  Input/Output: Atmp, Btmp
   *  Input: dlmax, drmax, dlmin, drmin
   */
  static void scale_Atmp_Btmp_dl_dr(Matrix& Atmp, Matrix& Btmp, const Vector& dlmax,
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
   * roles compared to scale_Atmp_Btmp_dl_dr.
   *
   *  Specifically:
   *  Xtmp(i, j) = Xtmp(i, j) / (drmax(i) * dlmax(j))
   *  Ytmp(i, j) = Ytmp(i, j) * (drmin(i) * dlmin(j))
   *
   *  Input/Output: Xtmp, Ytmp
   *  Input: drmax, dlmax, drmin, dlmin
   */
  static void scale_Xtmp_Ytmp_dl_dr(Matrix& Xtmp, Matrix& Ytmp, const Vector& drmax,
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
   *  to obtain equal-time Green's functions G(t,t)
   */
  static void compute_greens_00_bb(const Matrix& U, const Vector& S, const Matrix& V, Matrix& gtt) {
    // split S = Sbi^-1 * Ss
    Vector Sbi(S.size());
    Vector Ss(S.size());
    computeSbiSs(S, Sbi, Ss);

    // compute (1 + USV^T)^-1 in a stable manner
    // note that H is good conditioned, which only contains information of small
    // scale.
    Matrix H = Sbi.asDiagonal() * U.transpose() + Ss.asDiagonal() * V.transpose();

    // compute gtt using QR decomposition
    gtt = H.colPivHouseholderQr().solve(Sbi.asDiagonal() * U.transpose());
  }

  /*
   *  return (1 + USV^T)^-1 * USV^T, with method of QR decomposition
   *  to obtain time-displaced Green's functions G(beta, 0)
   */
  static void compute_greens_b0(const Matrix& U, const Vector& S, const Matrix& V, Matrix& gt0) {
    // split S = Sbi^-1 * Ss
    Vector Sbi(S.size());
    Vector Ss(S.size());
    computeSbiSs(S, Sbi, Ss);

    // compute (1 + USV^T)^-1 * USV^T in a stable manner
    // note that H is good conditioned, which only contains information of small
    // scale.
    Matrix H = Sbi.asDiagonal() * U.transpose() + Ss.asDiagonal() * V.transpose();

    // compute gtt using QR decomposition
    gt0 = H.colPivHouseholderQr().solve(Ss.asDiagonal() * V.transpose());
  }

  /*
   * Helper for common calculations in both compute_dynamic_greens and compute_equaltime_greens
   */
  static void compute_greens_function_common_part(const SvdStack& left, const SvdStack& right,
                                                  GreensWorkspace& ws) {
    auto& dlmax = ws.dlmax;
    auto& dlmin = ws.dlmin;
    auto& drmax = ws.drmax;
    auto& drmin = ws.drmin;
    auto& Atmp = ws.Atmp;
    auto& Btmp = ws.Btmp;
    auto& tmp = ws.tmp;

    const Matrix& ul = left.MatrixU();
    const Vector& dl = left.SingularValues();
    const Matrix& vl = left.MatrixV();
    const Matrix& ur = right.MatrixU();
    const Vector& dr = right.SingularValues();
    const Matrix& vr = right.MatrixV();

    div_dvec_max_min(dl, dlmax, dlmin);
    div_dvec_max_min(dr, drmax, drmin);

    Atmp.noalias() = ul.transpose() * ur;
    Btmp.noalias() = vl.transpose() * vr;

    scale_Atmp_Btmp_dl_dr(Atmp, Btmp, dlmax, drmax, dlmin, drmin);

    tmp.noalias() = Atmp + Btmp;

    auto& B_for_solve = ws.B_for_solve;
    B_for_solve.noalias() = ur * drmax.asDiagonal().inverse();

    Utils::LinearAlgebra::solve_X_times_A_eq_B(Atmp, tmp, B_for_solve, ws.qr_solver);
  }

  /*
   *  return (1 + left * right^T)^-1 in a stable manner, with method of MGS
   * factorization note: (1 + left * right^T)^-1 = (1 + (USV^T)_left *
   * (VSU^T)_right)^-1
   */
  static void compute_equaltime_greens(const SvdStack& left, const SvdStack& right, Matrix& gtt,
                                       GreensWorkspace& ws) {
    DQMC_ASSERT(left.MatDim() == right.MatDim());
    const int ndim = left.MatDim();
    ws.resize(ndim);

    if (left.empty()) {
      compute_greens_00_bb(right.MatrixV(), right.SingularValues(), right.MatrixU(), gtt);
      return;
    }
    if (right.empty()) {
      compute_greens_00_bb(left.MatrixU(), left.SingularValues(), left.MatrixV(), gtt);
      return;
    }

    compute_greens_function_common_part(left, right, ws);
    auto& Atmp = ws.Atmp;

    mult_v_invd_u(Atmp, ws.dlmax, left.MatrixU().transpose(), gtt);
  }

  /*
   *  return time-displaced Green's function in a stable manner,
   *  with the method of MGS factorization
   */
  static void compute_dynamic_greens(const SvdStack& left, const SvdStack& right, Matrix& gt0,
                                     Matrix& g0t, GreensWorkspace& ws) {
    DQMC_ASSERT(left.MatDim() == right.MatDim());
    const int ndim = left.MatDim();
    ws.resize(ndim);

    if (left.empty()) {
      compute_greens_00_bb(right.MatrixV(), right.SingularValues(), right.MatrixU(), gt0);

      g0t.noalias() = gt0;
      g0t.diagonal().array() -= 1.0;
      return;
    }

    if (right.empty()) {
      compute_greens_b0(left.MatrixU(), left.SingularValues(), left.MatrixV(), gt0);
      compute_greens_00_bb(left.MatrixU(), left.SingularValues(), left.MatrixV(), g0t);
      g0t *= -1.0;
      return;
    }

    compute_greens_function_common_part(left, right, ws);
    auto& Atmp = ws.Atmp;
    mult_v_d_u(Atmp, ws.dlmin, left.MatrixV().transpose(), gt0);

    auto& Xtmp = ws.Xtmp;
    auto& Ytmp = ws.Ytmp;
    auto& tmp = ws.tmp;

    const Matrix& ul = left.MatrixU();
    const Matrix& vl = left.MatrixV();
    const Matrix& ur = right.MatrixU();
    const Matrix& vr = right.MatrixV();

    Xtmp.noalias() = vr.transpose() * vl;
    Ytmp.noalias() = ur.transpose() * ul;

    scale_Xtmp_Ytmp_dl_dr(Xtmp, Ytmp, ws.drmax, ws.dlmax, ws.drmin, ws.dlmin);

    tmp.noalias() = Xtmp + Ytmp;

    auto& B_for_solve = ws.B_for_solve;
    B_for_solve.noalias() = (-vl) * ws.dlmax.asDiagonal().inverse();

    Utils::LinearAlgebra::solve_X_times_A_eq_B(Xtmp, tmp, B_for_solve, ws.qr_solver);
    mult_v_d_u(Xtmp, ws.drmin, ur.transpose(), g0t);
  }
};
}  // namespace Utils
