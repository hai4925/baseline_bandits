cmake_minimum_required (VERSION 2.8.8)
project (bandit_baseline CXX)

set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/Modules/")

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(Boost 1.54.0 COMPONENTS program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

set (CMAKE_BUILD_TYPE Release)
#SET (CMAKE_CXX_COMPILER "clang++")

if (CMAKE_COMPILER_IS_GNUCXX)
  set (CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wextra -Werror")  
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set (CMAKE_CXX_FLAGS "-std=c++11 -stdlib=libc++ -Wall -Werror")
endif()

add_executable(run_experiment run_experiment.cpp)
target_link_libraries(run_experiment ${Boost_LIBRARIES})
