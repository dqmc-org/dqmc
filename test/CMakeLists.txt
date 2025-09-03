add_executable(
  dqmc_tests
  placeholder_test.cpp
  measure/test_binning_analyzer.cpp
)

target_link_libraries(
  dqmc_tests
  dqmc
  gtest_main
  gtest
)

target_include_directories(
  dqmc_tests
  PRIVATE
  ${PROJECT_SOURCE_DIR}/include
)

include(GoogleTest)
gtest_discover_tests(dqmc_tests)
