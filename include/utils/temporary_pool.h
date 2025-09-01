#pragma once

#include <Eigen/Core>
#include <Eigen/QR>
#include <memory>

#ifdef DQMC_TEMPORARY_POOL_DEBUG
#include <iostream>
#endif

namespace Utils {

class TemporaryPool {
 public:
  using Matrix = Eigen::MatrixXd;
  using Vector = Eigen::VectorXd;
  using QRSolver = Eigen::ColPivHouseholderQR<Matrix>;

#ifdef DQMC_TEMPORARY_POOL_DEBUG
  struct TypeStats {
    size_t acquisitions = 0;
    size_t allocations = 0;
    size_t cache_hits = 0;
    size_t recycles = 0;
    size_t current_pool_size = 0;
    size_t pool_high_water_mark = 0;
  };

  struct DebugInfo {
    TypeStats matrix_stats;
    TypeStats vector_stats;
    TypeStats qr_solver_stats;

    void print(std::ostream& os = std::cout) const {
      auto print_stats = [&](const std::string& name, const TypeStats& stats) {
        os << "Acquisitions:" << stats.acquisitions << "\n"
           << "New Allocations:" << stats.allocations << "\n"
           << "Cache Hits:" << stats.cache_hits << "\n"
           << "Recycles:" << stats.recycles << "\n"
           << "Current Pool Size:" << stats.current_pool_size << "\n"
           << "Pool High-Water Mark:" << stats.pool_high_water_mark << "\n";
        if (stats.acquisitions > 0) {
          double hit_rate = static_cast<double>(stats.cache_hits) / stats.acquisitions * 100.0;
          os << "Cache Hit Rate:" << hit_rate << "%\n";
        }
        os << "\n";
      };

      print_stats("Matrix", matrix_stats);
      print_stats("Vector", vector_stats);
      print_stats("QR Solver", qr_solver_stats);
    }
  };
#endif

  template <typename T>
  struct PoolDeleter {
    TemporaryPool* pool_ptr;

    PoolDeleter(TemporaryPool* p = nullptr) : pool_ptr(p) {}

    PoolDeleter(const PoolDeleter&) = default;
    PoolDeleter& operator=(const PoolDeleter&) = default;

    void operator()(T* ptr) const {
      if (ptr == nullptr) {
        return;
      }
      if (pool_ptr) {
        pool_ptr->recycle(std::unique_ptr<T>(ptr));
      } else {
        delete ptr;
      }
    }
  };

  using MatrixPtr = std::unique_ptr<Eigen::MatrixXd, PoolDeleter<Eigen::MatrixXd>>;
  using VectorPtr = std::unique_ptr<Eigen::VectorXd, PoolDeleter<Eigen::VectorXd>>;
  using QRSolverPtr = std::unique_ptr<Eigen::ColPivHouseholderQR<Eigen::MatrixXd>,
                                      PoolDeleter<Eigen::ColPivHouseholderQR<Eigen::MatrixXd>>>;

  MatrixPtr acquire_matrix(int rows, int cols);
  VectorPtr acquire_vector(int size);
  QRSolverPtr acquire_qr_solver();

#ifdef DQMC_TEMPORARY_POOL_DEBUG
  const DebugInfo& debug_info() const { return m_debug_info; }
#endif

  TemporaryPool() = default;
  ~TemporaryPool() = default;
  TemporaryPool(const TemporaryPool&) = delete;
  TemporaryPool& operator=(const TemporaryPool&) = delete;
  TemporaryPool(TemporaryPool&&) = delete;
  TemporaryPool& operator=(TemporaryPool&&) = delete;

 private:
  friend struct PoolDeleter<Matrix>;
  friend struct PoolDeleter<Vector>;
  friend struct PoolDeleter<QRSolver>;

  void recycle(std::unique_ptr<Matrix> mat) {
    m_matrix_pool.push_back(std::move(mat));

#ifdef DQMC_TEMPORARY_POOL_DEBUG
    m_debug_info.matrix_stats.recycles++;
    m_debug_info.matrix_stats.current_pool_size = m_matrix_pool.size();
    if (m_debug_info.matrix_stats.current_pool_size >
        m_debug_info.matrix_stats.pool_high_water_mark) {
      m_debug_info.matrix_stats.pool_high_water_mark = m_debug_info.matrix_stats.current_pool_size;
    }
#endif
  }

  void recycle(std::unique_ptr<Vector> vec) {
    m_vector_pool.push_back(std::move(vec));

#ifdef DQMC_TEMPORARY_POOL_DEBUG
    m_debug_info.vector_stats.recycles++;
    m_debug_info.vector_stats.current_pool_size = m_vector_pool.size();
    if (m_debug_info.vector_stats.current_pool_size >
        m_debug_info.vector_stats.pool_high_water_mark) {
      m_debug_info.vector_stats.pool_high_water_mark = m_debug_info.vector_stats.current_pool_size;
    }
#endif
  }

  void recycle(std::unique_ptr<QRSolver> solver) {
    m_solver_pool.push_back(std::move(solver));

#ifdef DQMC_TEMPORARY_POOL_DEBUG
    m_debug_info.qr_solver_stats.recycles++;
    m_debug_info.qr_solver_stats.current_pool_size = m_solver_pool.size();
    if (m_debug_info.qr_solver_stats.current_pool_size >
        m_debug_info.qr_solver_stats.pool_high_water_mark) {
      m_debug_info.qr_solver_stats.pool_high_water_mark =
          m_debug_info.qr_solver_stats.current_pool_size;
    }
#endif
  }

  std::vector<std::unique_ptr<Matrix>> m_matrix_pool;
  std::vector<std::unique_ptr<Vector>> m_vector_pool;
  std::vector<std::unique_ptr<QRSolver>> m_solver_pool;

#ifdef DQMC_TEMPORARY_POOL_DEBUG
  DebugInfo m_debug_info;
#endif
};
}  // namespace Utils
