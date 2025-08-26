#pragma once

/*
 * SVD-based numerical stabilization for matrix chain multiplication
 *
 * This file provides classes for stable multiplication of long chains of dense
 * matrices using Singular Value Decomposition. This is essential in DQMC
 * algorithms where repeated matrix multiplications can lead to numerical
 * instabilities due to exponentially growing or decaying singular values.
 *
 * The approach decomposes each matrix product as A = U * S * V^T and carefully
 * manages the singular values to prevent overflow/underflow conditions.
 */

#include <Eigen/Core>
#include <iostream>
#include <vector>

namespace Utils {

/*
 * Container for the three components of an SVD decomposition: A = U * S * V^T
 *
 * For any matrix A, the SVD gives us:
 * - U: orthogonal matrix containing left singular vectors
 * - S: diagonal matrix of singular values (stored as a vector)
 * - V: orthogonal matrix containing right singular vectors
 *
 * This decomposition is numerically stable and allows us to identify
 * and handle ill-conditioned matrices in DQMC calculations.
 */
class SvdClass {
 private:
  using uMat = Eigen::MatrixXd;  // Left singular vectors matrix
  using sVec = Eigen::VectorXd;  // Singular values vector
  using vMat = Eigen::MatrixXd;  // Right singular vectors matrix

  uMat m_u_mat{};
  sVec m_s_vec{};
  vMat m_v_mat{};

 public:
  SvdClass() = default;

  // Initialize with specified matrix dimension
  explicit SvdClass(int dim) : m_u_mat(dim, dim), m_s_vec(dim), m_v_mat(dim, dim) {}

  // Access to the SVD components
  uMat& MatrixU() { return this->m_u_mat; }
  sVec& SingularValues() { return this->m_s_vec; }
  vMat& MatrixV() { return this->m_v_mat; }

  const uMat& MatrixU() const { return this->m_u_mat; }
  const sVec& SingularValues() const { return this->m_s_vec; }
  const vMat& MatrixV() const { return this->m_v_mat; }
};

/*
 * Numerically stable computation of matrix products A_n * A_{n-1} * ... * A_1 *
 * A_0
 *
 * Instead of computing the full matrix product directly (which can
 * overflow/underflow), this class maintains the SVD decomposition of the
 * accumulated product at each step.
 *
 * When a new matrix A_i is pushed, we compute:
 * SVD(A_i * U * S) = U' * S' * V'
 * and update the accumulated V matrix: V_total = V_total * V'
 *
 * This approach prevents numerical instabilities that commonly occur in DQMC
 * simulations where matrix elements can span many orders of magnitude.
 *
 * The final product can be reconstructed as: Product = U' * S' * V_total^T
 */
class SvdStack {
 private:
  using VecSvd = std::vector<SvdClass>;
  using Matrix = Eigen::MatrixXd;
  using Vector = Eigen::VectorXd;

  VecSvd m_stack{};                  // Stack of SVD decompositions
  int m_mat_dim{};                   // Dimension of matrices (assumed square)
  std::vector<Matrix> m_prefix_v{};  // Prefix multiplication of V matrices

 public:
  SvdStack() = default;

  // Pre-allocate space for a stack of given depth
  explicit SvdStack(int mat_dim, int stack_length);

  // Stack state queries
  bool empty() const;
  int MatDim() const;
  int StackLength() const;

  // Access to the current accumulated product's SVD components
  // These represent the decomposition of the entire matrix chain
  const Vector& SingularValues() const;  // Current singular values
  const Matrix& MatrixU() const;         // Current U matrix
  const Matrix& MatrixV() const;         // Accumulated V matrix across all operations

  // Reset the stack to empty state (memory remains allocated for reuse)
  void clear();

  // Core operations for building the matrix product
  // Push: multiply a new matrix from the left: Product = matrix * Product
  // This updates the SVD decomposition incrementally to maintain stability
  void push(const Matrix& matrix);

  // Pop: remove the most recently added matrix from the product
  void pop();
};
}  // namespace Utils
