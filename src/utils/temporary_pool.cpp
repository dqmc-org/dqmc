#include "utils/temporary_pool.h"

#include "utils/eigen_malloc_guard.h"

namespace Utils {

TemporaryPool::MatrixPtr TemporaryPool::acquire_matrix(int rows, int cols) {
  std::unique_ptr<Matrix> resource;
  if (m_matrix_pool.empty()) {
    EigenMallocGuard<true> alloc_guard;
    resource = std::make_unique<Matrix>(rows, cols);
  } else {
    resource = std::move(m_matrix_pool.back());
    m_matrix_pool.pop_back();
    resource->resize(rows, cols);
  }
  return MatrixPtr(resource.release(), PoolDeleter<Matrix>(this));
}

TemporaryPool::VectorPtr TemporaryPool::acquire_vector(int size) {
  std::unique_ptr<Vector> resource;
  if (m_vector_pool.empty()) {
    EigenMallocGuard<true> alloc_guard;
    resource = std::make_unique<Vector>(size);
  } else {
    resource = std::move(m_vector_pool.back());
    m_vector_pool.pop_back();
    resource->resize(size);
  }
  return VectorPtr(resource.release(), PoolDeleter<Vector>(this));
}

TemporaryPool::QRSolverPtr TemporaryPool::acquire_qr_solver() {
  std::unique_ptr<QRSolver> resource;
  if (m_solver_pool.empty()) {
    EigenMallocGuard<true> alloc_guard;
    resource = std::make_unique<QRSolver>();
  } else {
    resource = std::move(m_solver_pool.back());
    m_solver_pool.pop_back();
  }
  return QRSolverPtr(resource.release(), PoolDeleter<QRSolver>(this));
}
}  // namespace Utils
