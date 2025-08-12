#pragma once

/**
 *  This head file defines the interface Utils::NumericalStable class,
 *  which contains subroutines to help compute equal-time and
 *  time-displaced (dynamical) Greens function in a stable manner.
 */

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/QR>

#include "svd_stack.h"

namespace Utils {

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
  static void matrix_compare_error(const Matrix& umat, const Matrix& vmat,
                                   double& error) {
    assert(umat.rows() == vmat.rows());
    assert(umat.cols() == vmat.cols());
    assert(umat.rows() == umat.cols());
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
    assert(dvec.size() == dmax.size());
    assert(dvec.size() == dmin.size());
    assert((dvec.array() >= 0).all());
    dmax = dvec.cwiseMax(Vector::Ones(dvec.size()));
    dmin = dvec.cwiseMin(Vector::Ones(dvec.size()));
  }

  /*
   *  Subroutine to perform dense matrix * (diagonal matrix)^-1 * dense matrix
   *  Input: vmat, dvec, umat
   *  Output: zmat
   */
  static void mult_v_invd_u(const Matrix& vmat, const Vector& dvec,
                            const Matrix& umat, Matrix& zmat) {
    assert(vmat.cols() == umat.cols());
    assert(vmat.cols() == zmat.cols());
    assert(vmat.rows() == umat.rows());
    assert(vmat.rows() == zmat.rows());
    assert(vmat.rows() == vmat.cols());
    assert(vmat.cols() == dvec.size());
    zmat.noalias() = vmat * dvec.asDiagonal().inverse() * umat;
  }

  /*
   *  Subroutine to perform dense matrix * diagonal matrix * dense matrix
   *  Input: vmat, dvec, umat
   *  Output: zmat
   */
  static void mult_v_d_u(const Matrix& vmat, const Vector& dvec,
                         const Matrix& umat, Matrix& zmat) {
    assert(vmat.cols() == umat.cols());
    assert(vmat.cols() == zmat.cols());
    assert(vmat.rows() == umat.rows());
    assert(vmat.rows() == zmat.rows());
    assert(vmat.rows() == vmat.cols());
    assert(vmat.cols() == dvec.size());
    zmat.noalias() = vmat * dvec.asDiagonal() * umat;
  }

  /*
   * This function applies the following logic element-wise:
   * - If S(i) > 1: Sbi(i) = 1.0 / S(i), Ss(i) = 1.0
   * - If S(i) <= 1: Sbi(i) = 1.0, Ss(i) = S(i)
   *  Input: S
   *  Output: Sbi, Ss
   */
  static void computeSbiSs(const Eigen::VectorXd& S, Eigen::VectorXd& Sbi,
                           Eigen::VectorXd& Ss) {
    assert((S.array() >= 0).all());
    assert(Sbi.size() == S.size());
    assert(Ss.size() == S.size());
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
  static void scale_Atmp_Btmp_dl_dr(Matrix& Atmp, Matrix& Btmp,
                                    const Vector& dlmax, const Vector& drmax,
                                    const Vector& dlmin, const Vector& drmin) {
    assert(Atmp.rows() == dlmax.size());
    assert(Atmp.cols() == drmax.size());
    assert(Btmp.rows() == dlmin.size());
    assert(Btmp.cols() == drmin.size());
    assert(Atmp.rows() == Btmp.rows());
    assert(Atmp.cols() == Btmp.cols());

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
  static void scale_Xtmp_Ytmp_dl_dr(Matrix& Xtmp, Matrix& Ytmp,
                                    const Vector& drmax, const Vector& dlmax,
                                    const Vector& drmin, const Vector& dlmin) {
    assert(Xtmp.rows() == drmax.size());
    assert(Xtmp.cols() == dlmax.size());
    assert(Ytmp.rows() == drmin.size());
    assert(Ytmp.cols() == dlmin.size());
    assert(Xtmp.rows() == Ytmp.rows());
    assert(Xtmp.cols() == Ytmp.cols());

    Xtmp.array().colwise() /= drmax.array();
    Xtmp.array().rowwise() /= dlmax.array().transpose();

    Ytmp.array().colwise() *= drmin.array();
    Ytmp.array().rowwise() *= dlmin.array().transpose();
  }

  /*
   *  return (1 + USV^T)^-1, with method of QR decomposition
   *  to obtain equal-time Green's functions G(t,t)
   */
  static void compute_greens_00_bb(const Matrix& U, const Vector& S,
                                   const Matrix& V, Matrix& gtt) {
    // split S = Sbi^-1 * Ss
    Vector Sbi(S.size());
    Vector Ss(S.size());
    computeSbiSs(S, Sbi, Ss);

    // compute (1 + USV^T)^-1 in a stable manner
    // note that H is good conditioned, which only contains information of small
    // scale.
    Matrix H =
        Sbi.asDiagonal() * U.transpose() + Ss.asDiagonal() * V.transpose();

    // compute gtt using QR decomposition
    gtt = H.colPivHouseholderQr().solve(Sbi.asDiagonal() * U.transpose());
  }

  /*
   *  return (1 + USV^T)^-1 * USV^T, with method of QR decomposition
   *  to obtain time-displaced Green's functions G(beta, 0)
   */
  static void compute_greens_b0(const Matrix& U, const Vector& S,
                                const Matrix& V, Matrix& gt0) {
    // split S = Sbi^-1 * Ss
    Vector Sbi(S.size());
    Vector Ss(S.size());
    computeSbiSs(S, Sbi, Ss);

    // compute (1 + USV^T)^-1 * USV^T in a stable manner
    // note that H is good conditioned, which only contains information of small
    // scale.
    Matrix H =
        Sbi.asDiagonal() * U.transpose() + Ss.asDiagonal() * V.transpose();

    // compute gtt using QR decomposition
    gt0 = H.colPivHouseholderQr().solve(Ss.asDiagonal() * V.transpose());
  }

  /*
   *  return (1 + left * right^T)^-1 in a stable manner, with method of MGS
   * factorization note: (1 + left * right^T)^-1 = (1 + (USV^T)_left *
   * (VSU^T)_right)^-1
   */
  static void compute_equaltime_greens(const SvdStack& left,
                                       const SvdStack& right, Matrix& gtt) {
    assert(left.MatDim() == right.MatDim());
    const int ndim = left.MatDim();

    // at time slice t = 0
    if (left.empty()) {
      compute_greens_00_bb(right.MatrixV(), right.SingularValues(),
                           right.MatrixU(), gtt);
      return;
    }

    // at time slice t = nt (beta)
    if (right.empty()) {
      compute_greens_00_bb(left.MatrixU(), left.SingularValues(),
                           left.MatrixV(), gtt);
      return;
    }

    // local params
    const Matrix& ul = left.MatrixU();
    const Vector& dl = left.SingularValues();
    const Matrix& vl = left.MatrixV();
    const Matrix& ur = right.MatrixU();
    const Vector& dr = right.SingularValues();
    const Matrix& vr = right.MatrixV();

    Vector dlmax(dl.size()), dlmin(dl.size());
    Vector drmax(dr.size()), drmin(dr.size());

    Matrix Atmp(ndim, ndim), Btmp(ndim, ndim);
    Matrix tmp(ndim, ndim);

    // modified Gram-Schmidt (MGS) factorization
    // perfrom the breakups dr = drmax * drmin , dl = dlmax * dlmin
    div_dvec_max_min(dl, dlmax, dlmin);
    div_dvec_max_min(dr, drmax, drmin);

    // Atmp = ul^T * ur, Btmp = vl^T * vr
    Atmp = ul.transpose() * ur;
    Btmp = vl.transpose() * vr;

    // Atmp = dlmax^-1 * (ul^T * ur) * drmax^-1
    // Btmp = dlmin * (vl^T * vr) * drmin
    scale_Atmp_Btmp_dl_dr(Atmp, Btmp, dlmax, drmax, dlmin, drmin);

    tmp = Atmp + Btmp;
    mult_v_invd_u(ur, drmax, tmp.inverse(), Atmp);

    // finally obtain gtt
    mult_v_invd_u(Atmp, dlmax, ul.transpose(), gtt);
  }

  /*
   *  return time-displaced Green's function in a stable manner,
   *  with the method of MGS factorization
   */
  static void compute_dynamic_greens(const SvdStack& left,
                                     const SvdStack& right, Matrix& gt0,
                                     Matrix& g0t) {
    assert(left.MatDim() == right.MatDim());
    const int ndim = left.MatDim();

    // at time slice t = 0
    if (left.empty()) {
      // gt0 = gtt at t = 0
      compute_greens_00_bb(right.MatrixV(), right.SingularValues(),
                           right.MatrixU(), gt0);

      // g0t = - ( 1 - gtt ）at t = 0, and this is a natural extension of g0t
      // for t = 0. however from the physical point of view, g0t should
      // degenerate to gtt at t = 0,
      g0t = -(Matrix::Identity(ndim, ndim) - gt0);
      return;
    }

    // at time slice t = nt (beta)
    if (right.empty()) {
      // gt0 = ( 1 + B(beta, 0) )^-1 * B(beta, 0)
      compute_greens_b0(left.MatrixU(), left.SingularValues(), left.MatrixV(),
                        gt0);

      // g0t = -gtt at t = beta
      compute_greens_00_bb(left.MatrixU(), left.SingularValues(),
                           left.MatrixV(), g0t);
      g0t = -g0t;
      return;
    }

    // local params
    const Matrix& ul = left.MatrixU();
    const Vector& dl = left.SingularValues();
    const Matrix& vl = left.MatrixV();
    const Matrix& ur = right.MatrixU();
    const Vector& dr = right.SingularValues();
    const Matrix& vr = right.MatrixV();

    Vector dlmax(dl.size()), dlmin(dl.size());
    Vector drmax(dr.size()), drmin(dr.size());

    Matrix Atmp(ndim, ndim), Btmp(ndim, ndim);
    Matrix Xtmp(ndim, ndim), Ytmp(ndim, ndim);
    Matrix tmp(ndim, ndim);

    // modified Gram-Schmidt (MGS) factorization
    // perfrom the breakups dr = drmax * drmin , dl = dlmax * dlmin
    div_dvec_max_min(dl, dlmax, dlmin);
    div_dvec_max_min(dr, drmax, drmin);

    // compute gt0
    // Atmp = ul^T * ur, Btmp = vl^T * vr
    Atmp = ul.transpose() * ur;
    Btmp = vl.transpose() * vr;

    // Atmp = dlmax^-1 * (ul^T * ur) * drmax^-1
    // Btmp = dlmin * (vl^T * vr) * drmin
    scale_Atmp_Btmp_dl_dr(Atmp, Btmp, dlmax, drmax, dlmin, drmin);

    tmp = Atmp + Btmp;
    mult_v_invd_u(ur, drmax, tmp.inverse(), Atmp);
    mult_v_d_u(Atmp, dlmin, vl.transpose(), gt0);

    // compute g0t
    // Xtmp = vr^T * vl, Ytmp = ur^T * ul
    Xtmp = vr.transpose() * vl;
    Ytmp = ur.transpose() * ul;

    // Xtmp = drmax^-1 * (vr^T * vl) * dlmax^-1
    // Ytmp = drmin * (ur^T * ul) * dlmin
    scale_Xtmp_Ytmp_dl_dr(Xtmp, Ytmp, drmax, dlmax, drmin, dlmin);

    tmp = Xtmp + Ytmp;
    mult_v_invd_u(-vl, dlmax, tmp.inverse(), Xtmp);
    mult_v_d_u(Xtmp, drmin, ur.transpose(), g0t);
  }
};
}  // namespace Utils
