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
#include <iostream>
#include <stdexcept>

#include "lapacke.h"
#include "utils/assert.h"

namespace Utils {

// -------------------------------------  Utils::LinearAlgebra class
// ----------------------------------------
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
  static void dgesvd(const int& row, const int& col, const Eigen::MatrixXd& mat,
                     Eigen::MatrixXd& u, Eigen::VectorXd& s,
                     Eigen::MatrixXd& v) {
    DQMC_ASSERT(row == mat.rows());
    DQMC_ASSERT(col == mat.cols());
    // TODO: currently, the subroutine would fail
    // if the input matrix has different rows and columns
    DQMC_ASSERT(row == col);

    // matrix size
    int matrix_layout = LAPACK_ROW_MAJOR;
    lapack_int info{};
    lapack_int lda = row, ldu = row, ldvt = col;

    // local arrays
    std::vector<double> tmp_s(ldu * ldu);
    std::vector<double> tmp_u(ldu * row);
    std::vector<double> tmp_vt(ldvt * col);
    std::vector<double> mat_in(lda * col);
    std::vector<double> super_mat(ldu * lda);

    // convert eigen matrix to c-style array
    Eigen::Map<
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        mat_in.data(), lda, col) = mat;

    // compute SVD
    info = LAPACKE_dgesvd(matrix_layout, 'A', 'A', row, col, mat_in.data(), lda,
                          tmp_s.data(), tmp_u.data(), ldu, tmp_vt.data(), ldvt,
                          super_mat.data());

    // check for convergence
    if (info > 0) {
      throw std::runtime_error(
          "Utils::LinearAlgebra::mkl_lapack_dgesvd(): "
          "computing SVD failed to converge.");
    }

    // convert the results into Eigen style
    u = Eigen::Map<
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        tmp_u.data(), col, col);
    s = Eigen::Map<Eigen::Matrix<double, 1, Eigen::Dynamic, Eigen::RowMajor>>(
        tmp_s.data(), 1, col);
    v = Eigen::Map<
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
        tmp_vt.data(), row, row);
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
  static void dsyev(const int& size, const Eigen::MatrixXd& mat,
                    Eigen::VectorXd& s, Eigen::MatrixXd& t) {
    DQMC_ASSERT(mat.rows() == size);
    DQMC_ASSERT(mat.cols() == size);
    // make sure the input matrix is symmetric
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
