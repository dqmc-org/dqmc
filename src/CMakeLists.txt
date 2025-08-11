add_library(
  dqmc
  dqmc.cpp
  lattice/cubic.cpp
  lattice/square.cpp
  lattice/honeycomb.cpp
  lattice/lattice_base.cpp
  svd_stack.cpp
  checkerboard/cubic.cpp
  checkerboard/square.cpp
  dqmc_initializer.cpp
  dqmc_walker.cpp
  measure/measure_methods.cpp
  measure/measure_handler.cpp
  measure/observable_handler.cpp
  model/model_base.cpp
  model/attractive_hubbard.cpp
  model/repulsive_hubbard.cpp
)

target_include_directories(
  dqmc
  PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
)

target_link_libraries(
  dqmc
  PUBLIC
  BLAS::BLAS
  Eigen3::Eigen
  ${Boost_LIBRARIES}
)

add_executable(
  main
  dqmc_main.cpp
)

set_target_properties(
  main
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

target_link_libraries(
  main
  dqmc
)
