cmake_minimum_required (VERSION 2.8.8)
project (bandit_baseline CXX)

add_library (lib_gibbs_policy OBJECT gibbs_policy.cpp)

add_executable (test test.cpp
               $<TARGET_OBJECTS:lib_gibbs_policy>)

set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/Modules/")

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

set (CMAKE_BUILD_TYPE Release)
SET (CMAKE_CXX_COMPILER "clang++")
set (CMAKE_CXX_FLAGS "-std=c++11 -stdlib=libc++ -Wall")


