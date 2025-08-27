#include "svd_stack.h"

#include <cassert>
#include <cmath>
#include <iostream>

#include "utils/linear_algebra.hpp"

namespace Utils {

using VecSvd = std::vector<SvdClass>;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

/*
 * Implementation of SVD-based stable matrix multiplication
 *
 * The key insight is that instead of computing A_n * ... * A_1 * A_0 directly,
 * we maintain the SVD of the accumulated product and update it incrementally.
 * This prevents overflow/underflow that would occur with direct multiplication.
 */

// Constructor: pre-allocate SVD storage for the specified stack depth
// This avoids memory allocations during the simulation
SvdStack::SvdStack(int mat_dim, int stack_length) : m_mat_dim(mat_dim) {
  this->m_stack.reserve(stack_length);
  this->m_prefix_v.reserve(stack_length);
  this->m_temp_buffer = Matrix(mat_dim, mat_dim);
}

// Basic stack state queries
bool SvdStack::empty() const { return this->m_stack.empty(); }

int SvdStack::MatDim() const { return this->m_mat_dim; }

int SvdStack::StackLength() const { return this->m_stack.size(); }

// Reset stack to empty state without deallocating memory
void SvdStack::clear() {
  this->m_stack.clear();
  this->m_prefix_v.clear();
}

// Add a matrix to the product from the left: Product = matrix * Product
// This is the core of the numerical stabilization algorithm
void SvdStack::push(const Matrix& matrix) {
  DQMC_ASSERT(matrix.rows() == this->m_mat_dim && matrix.cols() == this->m_mat_dim);

  SvdClass svd(this->m_mat_dim);
  if (this->m_stack.empty()) {
    // First matrix: just compute its SVD directly
    Utils::LinearAlgebra::dgesvd(matrix, svd.MatrixU(), svd.SingularValues(), svd.MatrixV());
    this->m_prefix_v.push_back(svd.MatrixV());
  } else {
    // Subsequent matrices: multiply with existing decomposition
    // We compute matrix * U * S, then take SVD of the result
    // The order of operations is crucial to avoid numerical issues:
    // 1. First multiply matrix * U (preserves orthogonality)
    // 2. Then scale by singular values S
    // 3. Finally compute SVD of the combined result
    m_tmp_buffer.noalias() = (matrix * this->MatrixU()) * this->SingularValues().asDiagonal();
    Utils::LinearAlgebra::dgesvd(m_tmp_buffer, svd.MatrixU(), svd.SingularValues(), svd.MatrixV());
    this->m_prefix_v.push_back(this->m_prefix_v.back() * svd.MatrixV());
  }
  this->m_stack.push_back(std::move(svd));
}

// Remove the most recent matrix from the stack
// Memory is not deallocated, just stack depth is decreased
void SvdStack::pop() {
  DQMC_ASSERT(!this->m_stack.empty());
  this->m_stack.pop_back();
  this->m_prefix_v.pop_back();
}

// Get the current singular values of the accumulated product
const Vector& SvdStack::SingularValues() const {
  DQMC_ASSERT(!this->m_stack.empty());
  return this->m_stack.back().SingularValues();
}

// Get the current U matrix (left singular vectors) of the accumulated product
const Matrix& SvdStack::MatrixU() const {
  DQMC_ASSERT(!this->m_stack.empty());
  return this->m_stack.back().MatrixU();
}

// Get the accumulated V matrix across all operations in the stack
// This requires multiplying all V matrices from bottom to top of stack
// since each push() operation creates a new V that must be composed with
// previous ones. We avoid performing these multiplications by storing the
// partial left multiplications on a separated stack.
const Matrix& SvdStack::MatrixV() const {
  DQMC_ASSERT(!this->m_stack.empty());
  return this->m_prefix_v.back();
}

}  // namespace Utils
