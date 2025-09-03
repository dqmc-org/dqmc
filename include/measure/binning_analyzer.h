#pragma once

#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace Measure {

class BinningAnalyzer {
 public:
  BinningAnalyzer() = default;

  // Add a new data point (a block average) to the time series.
  void add_data_point(double value);

  // Perform the full binning analysis on the current time series.
  // This should be called periodically, not after every single data point.
  void update_analysis();

  // --- Accessors for results ---
  double get_mean() const { return m_mean; }
  double get_error() const { return m_error; }
  double get_autocorrelation_time() const { return m_tau; }
  int get_optimal_bin_size() const { return m_optimal_bin_size_blocks; }
  int get_num_data_points() const { return static_cast<int>(m_time_series.size()); }

  // Returns true if the relative error is below the target.
  // Handles cases where the mean is close to zero by using an absolute tolerance.
  bool is_converged(double target_relative_error, double absolute_tolerance = 1e-12) const;

 private:
  std::vector<double> m_time_series;

  // --- Analysis Results ---
  double m_mean = 0.0;
  double m_error = std::numeric_limits<double>::infinity();
  double m_tau = std::numeric_limits<double>::infinity();
  int m_optimal_bin_size_blocks = 0;

  // Internal helpers
  static double sample_mean(const std::vector<double>& v);
  static double sample_variance(const std::vector<double>& v, double mean);
};
}  // namespace Measure