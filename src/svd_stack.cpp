#include "svd_stack.h"

#include <cassert>
#include <cmath>

#include "utils/linear_algebra.hpp"

namespace Utils {

/*
 * Implementation of SVD-based stable matrix multiplication
 *
 * The key insight is that instead of computing A_n * ... * A_1 * A_0 directly,
 * we maintain the SVD of the accumulated product and update it incrementally.
 * This prevents overflow/underflow that would occur with direct multiplication.
 */

// Constructor: pre-allocate SVD storage for the specified stack depth
// This avoids memory allocations during the simulation
SVD_stack::SVD_stack(int dim, int stack_length) : m_dim(dim) {
  this->m_stack.reserve(stack_length);
  this->m_prefix_v.reserve(stack_length);
  this->m_tmp_buffer = Eigen::MatrixXd(dim, dim);
}

// Basic stack state queries
// Reset stack to empty state without deallocating memory
void SVD_stack::clear() {
  this->m_stack.clear();
  this->m_prefix_v.clear();
}

// Add a matrix to the product from the left: Product = matrix * Product
// This is the core of the numerical stabilization algorithm
void SVD_stack::push(const Eigen::MatrixXd& matrix) {
  DQMC_ASSERT(matrix.rows() == this->m_dim && matrix.cols() == this->m_dim);

  SVD svd(this->m_dim);
  if (this->m_stack.empty()) {
    // First matrix: just compute its SVD directly
    Utils::LinearAlgebra::dgesvd(matrix, svd.U(), svd.S(), svd.V());
    this->m_prefix_v.push_back(svd.V());
  } else {
    // Subsequent matrices: multiply with existing decomposition
    // We compute matrix * U * S, then take SVD of the result
    // The order of operations is crucial to avoid numerical issues:
    // 1. First multiply matrix * U (preserves orthogonality)
    // 2. Then scale by singular values S
    // 3. Finally compute SVD of the combined result
    m_tmp_buffer.noalias() = (matrix * this->U()) * this->S().asDiagonal();
    Utils::LinearAlgebra::dgesvd(m_tmp_buffer, svd.U(), svd.S(), svd.V());
    this->m_prefix_v.push_back(this->m_prefix_v.back() * svd.V());
  }
  this->m_stack.push_back(std::move(svd));
}

// Remove the most recent matrix from the stack
void SVD_stack::pop() {
  DQMC_ASSERT(!this->m_stack.empty());
  this->m_stack.pop_back();
  this->m_prefix_v.pop_back();
}

// Get the current singular values of the accumulated product
const Eigen::VectorXd& SVD_stack::S() const {
  DQMC_ASSERT(!this->m_stack.empty());
  return this->m_stack.back().S();
}

// Get the current U matrix (left singular vectors) of the accumulated product
const Eigen::MatrixXd& SVD_stack::U() const {
  DQMC_ASSERT(!this->m_stack.empty());
  return this->m_stack.back().U();
}

// Get the accumulated V matrix across all operations in the stack
// This requires multiplying all V matrices from bottom to top of stack
// since each push() operation creates a new V that must be composed with
// previous ones. We avoid performing these multiplications by storing the
// partial left multiplications on a separated stack.
const Eigen::MatrixXd& SVD_stack::V() const {
  DQMC_ASSERT(!this->m_stack.empty());
  return this->m_prefix_v.back();
}

}  // namespace Utils
