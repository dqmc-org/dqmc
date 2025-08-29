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

SVD_stack::SVD_stack(int dim, int stack_length) : m_dim(dim), m_current_size(0) {
  m_stack.resize(stack_length);
  for (auto& svd : m_stack) {
    svd.resize(dim);
  }

  m_prefix_V.resize(stack_length, Eigen::MatrixXd(dim, dim));
  m_tmp_buffer.resize(dim, dim);
}

void SVD_stack::clear() { this->m_current_size = 0; }

void SVD_stack::push(const Eigen::MatrixXd& matrix) {
  DQMC_ASSERT(matrix.rows() == this->m_dim && matrix.cols() == this->m_dim);
  DQMC_ASSERT(this->m_current_size < this->capacity() && "SVD_stack overflow");

  SVD& svd = this->m_stack[this->m_current_size];

  if (this->m_current_size == 0) {
    // First matrix: compute its SVD directly into the first slot.
    Utils::LinearAlgebra::dgesvd(matrix, svd.U(), svd.S(), svd.V());
    this->m_prefix_V[0].noalias() = svd.V();
  } else {
    // Subsequent matrices: multiply with the previous decomposition.
    m_tmp_buffer.noalias() = (matrix * this->U()) * this->S().asDiagonal();
    Utils::LinearAlgebra::dgesvd(m_tmp_buffer, svd.U(), svd.S(), svd.V());

    // Update the prefix V product
    this->m_prefix_V[this->m_current_size].noalias() =
        this->m_prefix_V[this->m_current_size - 1] * svd.V();
  }
  this->m_current_size++;
}

void SVD_stack::pop() {
  DQMC_ASSERT(this->m_current_size > 0 && "SVD_stack underflow");
  this->m_current_size--;
}

const Eigen::VectorXd& SVD_stack::S() const {
  DQMC_ASSERT(this->m_current_size > 0);
  return this->m_stack[this->m_current_size - 1].S();
}

const Eigen::MatrixXd& SVD_stack::U() const {
  DQMC_ASSERT(this->m_current_size > 0);
  return this->m_stack[this->m_current_size - 1].U();
}

const Eigen::MatrixXd& SVD_stack::V() const {
  DQMC_ASSERT(this->m_current_size > 0);
  return this->m_prefix_V[this->m_current_size - 1];
}

}  // namespace Utils
