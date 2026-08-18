cmake_minimum_required(VERSION 3.16)
project(dmpc CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(platoon platoon.cpp)
target_link_libraries(platoon PRIVATE Eigen3::Eigen)
