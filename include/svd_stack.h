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
class SVD {
 public:
  SVD() = default;

  explicit SVD(int dim) : m_u(dim, dim), m_s(dim), m_v(dim, dim) {}

  void resize(int dim) {
    this->m_u.resize(dim, dim);
    this->m_s.resize(dim);
    this->m_v.resize(dim, dim);
  }

  Eigen::MatrixXd& U() { return this->m_u; }
  const Eigen::MatrixXd& U() const { return this->m_u; }

  Eigen::VectorXd& S() { return this->m_s; }
  const Eigen::VectorXd& S() const { return this->m_s; }

  Eigen::MatrixXd& V() { return this->m_v; }
  const Eigen::MatrixXd& V() const { return this->m_v; }

 private:
  Eigen::MatrixXd m_u{};
  Eigen::VectorXd m_s{};
  Eigen::MatrixXd m_v{};
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
class SVD_stack {
 public:
  SVD_stack() = default;

  explicit SVD_stack(int dim, int stack_length);

  bool empty() const { return this->m_current_size == 0; }
  int dim() const { return this->m_dim; }
  int size() const { return this->m_current_size; }
  int capacity() const { return m_stack.size(); }

  void clear();

  // Push: multiply a new matrix from the left: Product = matrix * Product
  // This updates the SVD decomposition incrementally to maintain stability
  void push(const Eigen::MatrixXd& matrix);

  // Pop: remove the most recently added matrix from the product
  void pop();

  // Access to the current accumulated product's SVD components
  // These represent the decomposition of the entire matrix chain
  const Eigen::MatrixXd& U() const;  // Current U matrix
  const Eigen::VectorXd& S() const;  // Current singular values
  const Eigen::MatrixXd& V() const;  // Accumulated V matrix across all operations

 private:
  int m_dim{};
  int m_current_size{};                       // Current number of elements on the stack
  std::vector<SVD> m_stack{};                 // Pre-allocated pool of SVD decompositions
  std::vector<Eigen::MatrixXd> m_prefix_V{};  // Pre-allocated pool for V matrices
  Eigen::MatrixXd m_tmp_buffer{};             // Buffer for SVD input
  Eigen::MatrixXd m_prod_buffer{};            // Buffer for intermediate products
};
}  // namespace Utils
