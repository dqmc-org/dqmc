#include "measure/binning_analyzer.h"

#include <algorithm>
#include <cmath>
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
  const int N = static_cast<int>(
      m_time_series.size());  // Need sufficient data for analysis. The Wolff method needs N >> tau,
  // so we use a more conservative threshold than typical binning.
  if (N < 64) {
    m_mean = sample_mean(m_time_series);
    m_error = std::numeric_limits<double>::infinity();
    m_tau = std::numeric_limits<double>::infinity();
    m_optimal_window_size = 0;
    return;
  }

  m_mean = sample_mean(m_time_series);
  const double raw_var = sample_variance(m_time_series, m_mean);

  if (raw_var < 1e-12) {  // Handle constant data series
    m_error = 0.0;
    m_tau = 0.0;
    m_optimal_window_size = 0;
    return;
  }

  // --- Ulli Wolff's method for automatic windowing (hep-lat/0306017) ---

  // 1. Calculate the un-normalized autocorrelation function Gamma(t).
  // The window search can go up to W_max. Cap at N/4.
  const int W_max = N / 4;
  std::vector<double> gamma(W_max + 1);
  gamma[0] = raw_var;
  for (int t = 1; t <= W_max; ++t) {
    double sum = 0.0;
    for (int i = 0; i < N - t; ++i) {
      sum += (m_time_series[i] - m_mean) * (m_time_series[i + t] - m_mean);
    }
    gamma[t] = sum / static_cast<double>(N - t);  // Unbiased estimator for Gamma(t)
  }

  // 2. Find the optimal summation window W_opt.
  int W_opt = -1;
  const double S = 1.5;  // S = tau / tau_int, recommended value from the paper.
  double C_F_W = gamma[0];

  // Loop over possible window sizes W.
  for (int W = 1; W <= W_max; ++W) {
    C_F_W += 2.0 * gamma[W];

    // If C_F(W) becomes negative, the sum is dominated by noise.
    // The previous W was the last reliable one.
    if (C_F_W <= 0) {
      W_opt = W - 1;
      break;
    }

    double tau_int_W = C_F_W / (2.0 * gamma[0]);

    // The paper's condition requires an estimate of the exponential decay time tau,
    // which it calls tau_hat.
    double tau_hat_W;
    if (tau_int_W <= 0.5) {
      // If tau_int is small, correlations are negligible.
      // Setting tau_hat to a tiny value makes g(W) negative, terminating the search.
      tau_hat_W = 1e-9;
    } else {
      // Invert eq. (51) to get tau_hat from tau_int
      double x = 2.0 * tau_int_W;
      tau_hat_W = S / (2.0 * std::atanh(1.0 / x));
    }

    // Check the self-consistency condition (eq. 52).
    // The optimal W is where the systematic error exp(-W/tau)
    // becomes smaller than the statistical noise term tau/sqrt(W*N).
    double g_W = std::exp(-static_cast<double>(W) / tau_hat_W) -
                 tau_hat_W / std::sqrt(static_cast<double>(W) * static_cast<double>(N));

    if (g_W < 0) {
      W_opt = W;
      break;
    }
  }

  // Fallback: if the condition is never met, use the largest window.
  if (W_opt == -1) {
    W_opt = W_max;
  }
  m_optimal_window_size = W_opt;

  // 3. Calculate final error and tau_int using W_opt.
  double C_F_opt = gamma[0];
  for (int t = 1; t <= W_opt; ++t) {
    C_F_opt += 2.0 * gamma[t];
  }

  if (C_F_opt <= 0) {  // Safety check
    m_error = std::numeric_limits<double>::infinity();
    m_tau = std::numeric_limits<double>::infinity();
  } else {
    // Final error is sqrt(C_F / N)
    m_error = std::sqrt(C_F_opt / static_cast<double>(N));
    // Integrated autocorrelation time tau_int = C_F / (2 * Var)
    m_tau = C_F_opt / (2.0 * raw_var);
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
