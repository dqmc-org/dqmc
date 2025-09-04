#include "measure/binning_analyzer.h"

#include <algorithm>
#include <numeric>

namespace Measure {

void BinningAnalyzer::add_data_point(double value) { m_time_series.push_back(value); }

double BinningAnalyzer::sample_mean(const std::vector<double>& v) {
  if (v.empty()) return 0.0;
  double sum = std::accumulate(v.begin(), v.end(), 0.0);
  return sum / static_cast<double>(v.size());
}

double BinningAnalyzer::sample_variance(const std::vector<double>& v, double mean) {
  const int n = static_cast<int>(v.size());
  if (n <= 1) return 0.0;
  double accum = 0.0;
  for (double x : v) {
    double d = x - mean;
    accum += d * d;
  }
  return accum / static_cast<double>(n - 1);
}

void BinningAnalyzer::update_analysis() {
  const int N = static_cast<int>(m_time_series.size());
  if (N < 16) {  // Need sufficient data for analysis
    m_mean = sample_mean(m_time_series);
    m_error = std::numeric_limits<double>::infinity();
    m_tau = std::numeric_limits<double>::infinity();
    m_optimal_bin_size_blocks = 0;
    return;
  }

  m_mean = sample_mean(m_time_series);
  const double raw_var = sample_variance(m_time_series, m_mean);

  // Pre-allocate vectors to avoid heap allocation during analysis
  std::vector<int> bin_sizes;
  bin_sizes.reserve(20);  // Reasonable upper bound for log2 scaling
  for (int B = 1; B * 4 <= N;) {
    bin_sizes.push_back(B);
    int nextB = B * 3 / 2;
    if (nextB == B) nextB += 1;
    B = nextB;
  }
  if (bin_sizes.empty()) bin_sizes.push_back(1);

  std::vector<double> plateau_variances;
  plateau_variances.reserve(bin_sizes.size());
  std::vector<double> bin_mean_variances;
  bin_mean_variances.reserve(bin_sizes.size());

  for (int B : bin_sizes) {
    const int n_bins = N / B;
    std::vector<double> bin_means;
    bin_means.reserve(n_bins);
    for (int i = 0; i < n_bins; ++i) {
      double sum =
          std::accumulate(m_time_series.begin() + i * B, m_time_series.begin() + (i + 1) * B, 0.0);
      bin_means.push_back(sum / static_cast<double>(B));
    }
    double var_bmeans = sample_variance(bin_means, sample_mean(bin_means));
    bin_mean_variances.push_back(var_bmeans);
    plateau_variances.push_back(static_cast<double>(B) * var_bmeans);
  }

  // Heuristic plateau detection: Find the first bin size where the next 3
  // values are within 10%
  int plateau_idx = -1;
  for (size_t i = 0; i < plateau_variances.size(); ++i) {
    if (plateau_variances[i] <= 0.0) continue;
    bool is_stable = true;
    for (size_t j = i + 1; j < std::min(plateau_variances.size(), i + 4); ++j) {
      if (std::abs(plateau_variances[j] - plateau_variances[i]) / plateau_variances[i] > 0.05) {
        is_stable = false;
        break;
      }
    }
    if (is_stable) {
      plateau_idx = static_cast<int>(i);
      break;
    }
  }

  if (plateau_idx == -1) plateau_idx = bin_sizes.size() - 1;  // Fallback to largest

  m_optimal_bin_size_blocks = bin_sizes[plateau_idx];
  const int n_bins_final = N / m_optimal_bin_size_blocks;
  m_error = std::sqrt(bin_mean_variances[plateau_idx] / n_bins_final);

  if (raw_var > 1e-12) {
    m_tau = 0.5 * (plateau_variances[plateau_idx] / raw_var - 1.0);
  } else {
    m_tau = 0.0;
  }
}

bool BinningAnalyzer::is_converged(double target_relative_error, double absolute_tolerance) const {
  if (!std::isfinite(m_error) || !std::isfinite(m_mean)) return false;
  if (std::abs(m_mean) < absolute_tolerance) {
    return m_error < (target_relative_error * absolute_tolerance);
  }
  return (m_error / std::abs(m_mean)) < target_relative_error;
}
}  // namespace Measure
