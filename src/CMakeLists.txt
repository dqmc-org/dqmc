add_library(
  dqmc
  dqmc.cpp
  lattice/cubic.cpp
  lattice/square.cpp
  lattice/chain.cpp
  lattice/honeycomb.cpp
  lattice/lattice_base.cpp
  svd_stack.cpp
  checkerboard/cubic.cpp
  checkerboard/square.cpp
  walker.cpp
  measure/measure_methods.cpp
  measure/measure_handler.cpp
  measure/observable_handler.cpp
  model/model_base.cpp
  model/attractive_hubbard.cpp
  model/repulsive_hubbard.cpp
  utils/assert.cpp
  utils/temporary_pool.cpp
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
  lapacke
)

target_compile_definitions(
  dqmc
  PUBLIC
  EIGEN_USE_BLAS
  EIGEN_USE_LAPACKE
  EIGEN_RUNTIME_NO_MALLOC
)

add_executable(
  main
  main.cpp
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
