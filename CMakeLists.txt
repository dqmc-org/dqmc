cmake_minimum_required(VERSION 3.21)
project(dqmc)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(BLAS REQUIRED)
find_package(Eigen3 3.4.0 REQUIRED)
find_package(Boost 1.83.0 COMPONENTS program_options REQUIRED)

add_subdirectory(third-party)
add_subdirectory(src)

if (BUILD_TESTING)
  add_subdirectory(test)
endif ()

