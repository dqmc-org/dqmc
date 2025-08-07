#ifndef UTILS_LINEAR_ALGEBRA_HPP
#define UTILS_LINEAR_ALGEBRA_HPP
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
    assert(row == mat.rows());
    assert(col == mat.cols());
    // TODO: currently, the subroutine would fail
    // if the input matrix has different rows and columns
    assert(row == col);

    // use Eigen's JacobiSVD for SVD decomposition
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        mat, Eigen::ComputeFullU | Eigen::ComputeFullV);

    if (svd.info() != Eigen::Success) {
      std::cerr << "Utils::LinearAlgebra::dgesvd(): "
                << "the algorithm computing SVD failed to converge."
                << std::endl;
      exit(1);
    }

    // get SVD components: A = U * S * V^T
    // Note: the function comment says V is returned, not V^T
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
  static void dsyev(const int& size, const Eigen::MatrixXd& mat,
                    Eigen::VectorXd& s, Eigen::MatrixXd& t) {
    assert(mat.rows() == size);
    assert(mat.cols() == size);
    // make sure the input matrix is symmetric
    // assert(mat.isApprox(mat.transpose(), 1e-12));

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(mat);

    if (solver.info() != Eigen::Success) {
      std::cerr << "Utils::LinearAlgebra::dsyev(): "
                << "the algorithm failed to compute eigenvalues." << std::endl;
      exit(1);
    }

    s = solver.eigenvalues();
    t = solver.eigenvectors();
  }
};

}  // namespace Utils

#endif  // UTILS_LINEAR_ALGEBRA_HPP
