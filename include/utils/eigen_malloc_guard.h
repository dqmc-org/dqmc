#pragma once

#ifdef EIGEN_RUNTIME_NO_MALLOC

#include <eigen3/Eigen/Core>

/**
 * @brief An RAII-style guard to disallow Eigen memory allocations in a specific scope.
 *
 * When an object of this class is created, it calls Eigen::internal::set_is_malloc_allowed(false).
 * When it is destroyed (at the end of its scope), it automatically calls
 * Eigen::internal::set_is_malloc_allowed(true), restoring the previous state.
 *
 * This is exception-safe and prevents forgetting to re-enable allocations.
 */
template <bool Enable>
class EigenMallocGuard {
 public:
  EigenMallocGuard() { Eigen::internal::set_is_malloc_allowed(Enable); }
  ~EigenMallocGuard() { Eigen::internal::set_is_malloc_allowed(!Enable); }

  EigenMallocGuard(const EigenMallocGuard&) = delete;
  EigenMallocGuard& operator=(const EigenMallocGuard&) = delete;
  EigenMallocGuard(EigenMallocGuard&&) = delete;
  EigenMallocGuard& operator=(EigenMallocGuard&&) = delete;
};
#else
template <bool Enable = true>
class EigenMallocGuard {
 public:
  EigenMallocGuard() = default;
  ~EigenMallocGuard() = default;

  EigenMallocGuard(const EigenMallocGuard&) = delete;
  EigenMallocGuard& operator=(const EigenMallocGuard&) = delete;
  EigenMallocGuard(EigenMallocGuard&&) = delete;
  EigenMallocGuard& operator=(EigenMallocGuard&&) = delete;
};
#endif  // EIGEN_RUNTIME_NO_MALLOC
