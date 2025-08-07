cmake_minimum_required(VERSION 3.21)
project(dqmc)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(Eigen3 3.4.0 REQUIRED)
find_package(Boost 1.71.0 COMPONENTS program_options mpi serialization REQUIRED)

add_subdirectory(src)

