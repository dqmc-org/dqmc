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
SvdStack::SvdStack(int mat_dim, int stack_length)
    : m_mat_dim(mat_dim),
      m_tmp_matrix(mat_dim, mat_dim),
      m_cached_v_matrix(mat_dim, mat_dim) {
  this->m_stack.resize(stack_length);
}

// Basic stack state queries
bool SvdStack::empty() const { return this->m_stack_length == 0; }

int SvdStack::MatDim() const { return this->m_mat_dim; }

int SvdStack::StackLength() const { return this->m_stack_length; }

// Reset stack to empty state without deallocating memory
void SvdStack::clear() { this->m_stack_length = 0; }

// Add a matrix to the product from the left: Product = matrix * Product
// This is the core of the numerical stabilization algorithm
void SvdStack::push(const Matrix& matrix) {
  assert(matrix.rows() == this->m_mat_dim && matrix.cols() == this->m_mat_dim);
  assert(this->m_stack_length < (int)this->m_stack.size());

  if (this->m_stack_length == 0) {
    // First matrix: just compute its SVD directly
    Utils::LinearAlgebra::dgesvd(
        this->m_mat_dim, this->m_mat_dim, matrix,
        this->m_stack[this->m_stack_length].MatrixU(),
        this->m_stack[this->m_stack_length].SingularValues(),
        this->m_stack[this->m_stack_length].MatrixV());
    this->m_cached_v_matrix = this->m_stack[this->m_stack_length].MatrixV();
  } else {
    // Subsequent matrices: multiply with existing decomposition
    // We compute matrix * U * S, then take SVD of the result
    // The order of operations is crucial to avoid numerical issues:
    // 1. First multiply matrix * U (preserves orthogonality)
    // 2. Then scale by singular values S
    // 3. Finally compute SVD of the combined result
    this->m_tmp_matrix =
        (matrix * this->MatrixU()) * this->SingularValues().asDiagonal();
    Utils::LinearAlgebra::dgesvd(
        this->m_mat_dim, this->m_mat_dim, this->m_tmp_matrix,
        this->m_stack[this->m_stack_length].MatrixU(),
        this->m_stack[this->m_stack_length].SingularValues(),
        this->m_stack[this->m_stack_length].MatrixV());
    this->m_cached_v_matrix *= this->m_stack[this->m_stack_length].MatrixV();
  }
  this->m_stack_length += 1;
}

// Remove the most recent matrix from the stack
// Memory is not deallocated, just stack depth is decreased
void SvdStack::pop() {
  assert(this->m_stack_length > 0);
  this->m_stack_length -= 1;
  is_v_matrix_cached = false;
}

// Get the current singular values of the accumulated product
const Vector SvdStack::SingularValues() {
  assert(this->m_stack_length > 0);
  return this->m_stack[this->m_stack_length - 1].SingularValues();
}

// Get the current U matrix (left singular vectors) of the accumulated product
const Matrix SvdStack::MatrixU() {
  assert(this->m_stack_length > 0);
  return this->m_stack[this->m_stack_length - 1].MatrixU();
}

// Get the accumulated V matrix across all operations in the stack
// This requires multiplying all V matrices from bottom to top of stack
// since each push() operation creates a new V that must be composed with
// previous ones
const Matrix SvdStack::MatrixV() {
  if (is_v_matrix_cached) {
    return this->m_cached_v_matrix;
  }

  assert(this->m_stack_length > 0);

  Matrix r = this->m_stack[0].MatrixV();
  for (int i = 1; i < this->m_stack_length; ++i) {
    r = r * this->m_stack[i].MatrixV();
  }

  this->m_cached_v_matrix = std::move(r);
  is_v_matrix_cached = true;

  return this->m_cached_v_matrix;
}

}  // namespace Utils
