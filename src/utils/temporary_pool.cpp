#include "utils/temporary_pool.h"

#include "utils/eigen_malloc_guard.h"

namespace Utils {

TemporaryPool::MatrixPtr TemporaryPool::acquire_matrix(int rows, int cols) {
#ifdef DQMC_TEMPORARY_POOL_DEBUG
  m_debug_info.matrix_stats.acquisitions++;
#endif

  std::unique_ptr<Matrix> resource;
  if (m_matrix_pool.empty()) {
    EigenMallocGuard<true> alloc_guard;
    resource = std::make_unique<Matrix>(rows, cols);

#ifdef DQMC_TEMPORARY_POOL_DEBUG
    m_debug_info.matrix_stats.allocations++;
#endif
  } else {
    for (auto& elm : m_matrix_pool) {
      if (elm->rows() == rows && elm->cols() == cols) {
        std::swap(elm, m_matrix_pool.back());
        break;
      }
    }

    resource = std::move(m_matrix_pool.back());
    m_matrix_pool.pop_back();
    resource->resize(rows, cols);

#ifdef DQMC_TEMPORARY_POOL_DEBUG
    m_debug_info.matrix_stats.cache_hits++;
    m_debug_info.matrix_stats.current_pool_size = m_matrix_pool.size();
#endif
  }

  return MatrixPtr(resource.release(), PoolDeleter<Matrix>(this));
}

TemporaryPool::VectorPtr TemporaryPool::acquire_vector(int size) {
#ifdef DQMC_TEMPORARY_POOL_DEBUG
  m_debug_info.vector_stats.acquisitions++;
#endif

  std::unique_ptr<Vector> resource;
  if (m_vector_pool.empty()) {
    EigenMallocGuard<true> alloc_guard;
    resource = std::make_unique<Vector>(size);

#ifdef DQMC_TEMPORARY_POOL_DEBUG
    m_debug_info.vector_stats.allocations++;
#endif
  } else {
    for (auto& elm : m_vector_pool) {
      if (elm->size() == size) {
        std::swap(elm, m_vector_pool.back());
        break;
      }
    }

    resource = std::move(m_vector_pool.back());
    m_vector_pool.pop_back();
    resource->resize(size);

#ifdef DQMC_TEMPORARY_POOL_DEBUG
    m_debug_info.vector_stats.cache_hits++;
    m_debug_info.vector_stats.current_pool_size = m_vector_pool.size();
#endif
  }
  return VectorPtr(resource.release(), PoolDeleter<Vector>(this));
}

TemporaryPool::QRSolverPtr TemporaryPool::acquire_qr_solver() {
#ifdef TEMPORARY_POOL_DEBUG
  m_debug_info.qr_solver_stats.acquisitions++;
#endif

  std::unique_ptr<QRSolver> resource;
  if (m_solver_pool.empty()) {
    EigenMallocGuard<true> alloc_guard;
    resource = std::make_unique<QRSolver>();

#ifdef DQMC_TEMPORARY_POOL_DEBUG
    m_debug_info.qr_solver_stats.allocations++;
#endif
  } else {
    resource = std::move(m_solver_pool.back());
    m_solver_pool.pop_back();

#ifdef DQMC_TEMPORARY_POOL_DEBUG
    m_debug_info.qr_solver_stats.cache_hits++;
    m_debug_info.qr_solver_stats.current_pool_size = m_solver_pool.size();
#endif
  }
  return QRSolverPtr(resource.release(), PoolDeleter<QRSolver>(this));
}
}  // namespace Utils
