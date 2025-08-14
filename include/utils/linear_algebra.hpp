#pragma once

/**
 *  This source file includes some diagonalizing tools with C++/Eigen interface
 *  for diagonalizing real matrices.
 *  including:
 *    1. generalized SVD decomposition for arbitrary M * N matrices
 *    2. optimized diagonalizing mechanism for N * N real symmetric matrix
 *  The calculation accuracy and efficiency are guaranteed.
 */

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <stdexcept>

#include "utils/assert.h"

namespace Utils {

class LinearAlgebra {
 public:
  /**
   *  SVD decomposition of arbitrary M * N real matrix:
   *       A  ->  U * S * V^T
   *  Remind that V is returned in this subroutine, not V transpose.
   *
   *  @param row -> number of rows.
   *  @param col -> number of cols.
   *  @param mat -> arbitrary `row` * `col` real matrix to be solved.
   *  @param u -> u matrix of type Eigen::MatrixXd, `row` * `row`.
   *  @param s -> eigenvalues s of type Eigen::VectorXd, descending sorted.
   *  @param v -> v matrix of type Eigen::MatrixXd, `col` * `col`.
   */
  static void dgesvd(const Eigen::MatrixXd& mat, Eigen::MatrixXd& u,
                     Eigen::VectorXd& s, Eigen::MatrixXd& v) {
    // BUG: Eigen Jacobi is not compatible with LAPACK dgesvd, make sure to
    // enable EIGEN_USE_BLAS and EIGEN_USE_LAPACKE. BDCSVD lowers to JacobiSVD
    // for sizes below or equal to 16. For more details:
    // https://eigen.tuxfamily.org/dox-devel/TopicUsingBlasLapack.html

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        mat, Eigen::ComputeFullU | Eigen::ComputeFullV);

    if (svd.info() != Eigen::Success) {
      throw std::runtime_error(
          "Utils::LinearAlgebra::dgesvd(): "
          "SVD algorithm failed.");
    }

    u = svd.matrixU();
    s = svd.singularValues();
    v = svd.matrixV();
  }

  /**
   *  Calculate eigenvalues and eigenstates given an arbitrary N * N real
   * symmetric matrix, A  ->  T^dagger * S * T where T is the
   * rotation matrix, which is orthogonal; S is a diagonal matrix with
   * eigenvalues being diagonal elements.
   *
   *  @param size -> number of rows/cols.
   *  @param mat -> arbitrary `size` * `size` real symmetric matrix to be
   * solved.
   *  @param s -> diagonal eigen matrix.
   *  @param t -> rotation matrix, whose columns are corresponding eigenstates.
   */
  static void dsyev(const Eigen::MatrixXd& mat, Eigen::VectorXd& s,
                    Eigen::MatrixXd& t) {
    DQMC_ASSERT(mat.rows() == mat.cols());
    DQMC_ASSERT(mat.isApprox(mat.transpose(), 1e-12));

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(mat);

    if (solver.info() != Eigen::Success) {
      throw std::runtime_error(
          "Utils::LinearAlgebra::dsyev(): "
          "failed to compute eigenvalues.");
    }

    s = solver.eigenvalues();
    t = solver.eigenvectors();
  }
};
}  // namespace Utils
