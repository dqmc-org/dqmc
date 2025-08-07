cmake_minimum_required(VERSION 3.21)
project(dqmc)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(MKL MODULE REQUIRED)
find_package(MPI MODULE REQUIRED)
find_package(Eigen3 MODULE 3.4.0 REQUIRED)
find_package(Boost MODULE 1.71.0 COMPONENTS program_options mpi serialization REQUIRED)

add_executable(
  main
  src/dqmc_main.cpp
)

target_link_libraries(
  main
  dqmc
)

add_library(
  dqmc
  src/dqmc.cpp
  src/lattice/cubic.cpp
  src/lattice/square.cpp
  src/lattice/honeycomb.cpp
  src/lattice/lattice_base.cpp
  src/svd_stack.cpp
  src/random.cpp
  src/checkerboard/cubic.cpp
  src/checkerboard/square.cpp
  src/dqmc_initializer.cpp
  src/dqmc_walker.cpp
  src/measure/measure_methods.cpp
  src/measure/measure_handler.cpp
  src/measure/observable_handler.cpp
  src/model/model_base.cpp
  src/model/attractive_hubbard.cpp
  src/model/repulsive_hubbard.cpp
)

target_include_directories(
  dqmc
  PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${MKL_INCLUDE_DIR}
  ${MPI_CXX_INCLUDE_PATH}
  ${EIGEN3_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
)

target_link_libraries(
  dqmc
  PUBLIC
  ${MKL_LIBRARIES}
  MPI::MPI_CXX
  Eigen3::Eigen
  ${Boost_LIBRARIES}
)

