#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "measure/binning_analyzer.h"

namespace {

// Helper function to generate a time series from an AR(1) process.
// The process is defined as: x_t = target_mean + alpha * (x_{t-1} - target_mean) + noise
std::vector<double> generate_ar1_series(size_t n, double alpha, double target_mean,
                                        double noise_stddev) {
  // Set up a random number generator for the noise term
  std::mt19937 gen(12345);  // Use a fixed seed for reproducibility
  std::normal_distribution<> noise_dist(0.0, noise_stddev);

  std::vector<double> series;
  series.reserve(n);

  // Start the series at the target mean
  double previous_value = target_mean;

  for (size_t i = 0; i < n; ++i) {
    double noise = noise_dist(gen);
    double current_value = target_mean + alpha * (previous_value - target_mean) + noise;
    series.push_back(current_value);
    previous_value = current_value;
  }
  return series;
}

}  // namespace

class BinningAnalyzerTest : public ::testing::Test {
 protected:
  Measure::BinningAnalyzer analyzer;
};

TEST_F(BinningAnalyzerTest, InitialState) {
  EXPECT_EQ(analyzer.get_num_data_points(), 0);
  EXPECT_EQ(analyzer.get_mean(), 0.0);
  EXPECT_TRUE(std::isinf(analyzer.get_error()));
  EXPECT_TRUE(std::isinf(analyzer.get_autocorrelation_time()));
  EXPECT_EQ(analyzer.get_optimal_bin_size(), 0);
  EXPECT_FALSE(analyzer.is_converged(0.01));
}

TEST_F(BinningAnalyzerTest, InsufficientData) {
  std::vector<double> data = {1.0, 2.0, 1.5, 2.5, 1.2, 2.2, 1.8, 2.8, 1.1, 2.1};  // 10 points
  for (double val : data) {
    analyzer.add_data_point(val);
  }

  analyzer.update_analysis();

  EXPECT_EQ(analyzer.get_num_data_points(), 10);
  double expected_mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
  EXPECT_DOUBLE_EQ(analyzer.get_mean(), expected_mean);

  // With < 16 points, analysis should not produce finite error/tau
  EXPECT_TRUE(std::isinf(analyzer.get_error()));
  EXPECT_TRUE(std::isinf(analyzer.get_autocorrelation_time()));
  EXPECT_EQ(analyzer.get_optimal_bin_size(), 0);
}

TEST_F(BinningAnalyzerTest, UncorrelatedDataWhiteNoise) {
  // For uncorrelated data (alpha = 0), tau should be 0.
  const double alpha = 0.0;
  const double theoretical_tau = 0.0;
  const double target_mean = 10.0;
  const size_t num_points = 16384;  // 2^14

  auto series = generate_ar1_series(num_points, alpha, target_mean, 2.0);
  for (double val : series) {
    analyzer.add_data_point(val);
  }

  analyzer.update_analysis();

  EXPECT_EQ(analyzer.get_num_data_points(), num_points);
  // The sample mean should be close to the target mean.
  EXPECT_NEAR(analyzer.get_mean(), target_mean, 0.05);

  // For uncorrelated data, the optimal bin size is 1 and tau is 0.
  // The plateau finding heuristic might occasionally pick a larger bin size
  // due to statistical fluctuations, but tau should remain very small.
  EXPECT_NEAR(analyzer.get_autocorrelation_time(), theoretical_tau, 0.1);

  // The error should be close to the standard error of the mean for uncorrelated data.
  double sum_sq_diff = 0.0;
  for (double val : series) {
    sum_sq_diff += (val - analyzer.get_mean()) * (val - analyzer.get_mean());
  }
  double sample_variance = sum_sq_diff / (num_points - 1);
  double expected_error = std::sqrt(sample_variance / num_points);

  EXPECT_NEAR(analyzer.get_error(), expected_error, expected_error * 0.15);  // 15% tolerance
}

TEST_F(BinningAnalyzerTest, ModeratelyCorrelatedData) {
  // AR(1) with alpha = 0.5 -> theoretical tau = 0.5 / (1 - 0.5) = 1.0
  const double alpha = 0.5;
  const double theoretical_tau = 1.0;
  const double target_mean = -5.0;
  const size_t num_points = 16384;  // 2^14

  auto series = generate_ar1_series(num_points, alpha, target_mean, 1.0);
  for (double val : series) {
    analyzer.add_data_point(val);
  }

  analyzer.update_analysis();

  EXPECT_EQ(analyzer.get_num_data_points(), num_points);
  EXPECT_NEAR(analyzer.get_mean(), target_mean, 0.1);

  // The estimated tau should be close to the theoretical value.
  // Statistical estimates require some tolerance.
  EXPECT_NEAR(analyzer.get_autocorrelation_time(), theoretical_tau, 0.2);

  // For tau=1, a common rule of thumb is that the optimal bin size should be > 2*tau.
  EXPECT_GE(analyzer.get_optimal_bin_size(), 2 * theoretical_tau);
}

TEST_F(BinningAnalyzerTest, StronglyCorrelatedData) {
  // AR(1) with alpha = 0.9 -> theoretical tau = 0.9 / (1 - 0.9) = 9.0
  // This requires a much longer time series to get a reliable estimate.
  const double alpha = 0.9;
  const double theoretical_tau = 9.0;
  const double target_mean = 25.0;
  const size_t num_points = 65536;  // 2^16

  auto series = generate_ar1_series(num_points, alpha, target_mean, 5.0);
  for (double val : series) {
    analyzer.add_data_point(val);
  }

  analyzer.update_analysis();

  EXPECT_EQ(analyzer.get_num_data_points(), num_points);
  EXPECT_NEAR(analyzer.get_mean(), target_mean, 0.5);

  // The estimated tau for highly correlated data has higher variance.
  // We use a larger tolerance.
  EXPECT_NEAR(analyzer.get_autocorrelation_time(), theoretical_tau, 3.5);

  // Optimal bin size should be significantly larger now.
  EXPECT_GE(analyzer.get_optimal_bin_size(), 2 * theoretical_tau);
}

TEST_F(BinningAnalyzerTest, ConvergenceCheck) {
  // Generate some data to get a non-infinite error
  auto series = generate_ar1_series(1024, 0.5, 50.0, 2.0);
  for (double val : series) {
    analyzer.add_data_point(val);
  }
  analyzer.update_analysis();

  // The analyzer should now have a finite mean and error
  ASSERT_TRUE(std::isfinite(analyzer.get_error()));
  ASSERT_TRUE(std::isfinite(analyzer.get_mean()));
  ASSERT_GT(std::abs(analyzer.get_mean()), 1e-9);

  double relative_error = analyzer.get_error() / std::abs(analyzer.get_mean());

  // Should be converged if target is larger than actual relative error
  EXPECT_TRUE(analyzer.is_converged(relative_error * 1.01));
  EXPECT_TRUE(analyzer.is_converged(relative_error + 1e-6));

  // Should NOT be converged if target is smaller
  EXPECT_FALSE(analyzer.is_converged(relative_error * 0.99));
  EXPECT_FALSE(analyzer.is_converged(relative_error - 1e-6));
}

TEST_F(BinningAnalyzerTest, ConvergenceCheckNearZeroMean) {
  // Generate data with a mean very close to zero
  auto series = generate_ar1_series(1024, 0.2, 0.0, 1e-13);
  for (double val : series) {
    analyzer.add_data_point(val);
  }
  analyzer.update_analysis();

  ASSERT_TRUE(std::isfinite(analyzer.get_error()));
  ASSERT_TRUE(std::isfinite(analyzer.get_mean()));

  // Mean should be small enough to trigger the absolute tolerance path
  const double abs_tol = 1e-12;
  ASSERT_LT(std::abs(analyzer.get_mean()), abs_tol);

  double error = analyzer.get_error();
  double target_rel_error = 0.01;

  // The convergence condition becomes: error < target_rel_error * abs_tol
  bool expected_convergence = (error < target_rel_error * abs_tol);

  EXPECT_EQ(analyzer.is_converged(target_rel_error, abs_tol), expected_convergence);
}