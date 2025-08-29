#pragma once

#include <Eigen/Core>
#include <Eigen/QR>
#include <memory>

namespace Utils {

class TemporaryPool {
 public:
  using Matrix = Eigen::MatrixXd;
  using Vector = Eigen::VectorXd;
  using QRSolver = Eigen::ColPivHouseholderQR<Matrix>;

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

  void recycle(std::unique_ptr<Matrix> mat) { m_matrix_pool.push_back(std::move(mat)); }
  void recycle(std::unique_ptr<Vector> vec) { m_vector_pool.push_back(std::move(vec)); }
  void recycle(std::unique_ptr<QRSolver> solver) { m_solver_pool.push_back(std::move(solver)); }

  std::vector<std::unique_ptr<Matrix>> m_matrix_pool;
  std::vector<std::unique_ptr<Vector>> m_vector_pool;
  std::vector<std::unique_ptr<QRSolver>> m_solver_pool;
};
}  // namespace Utils
