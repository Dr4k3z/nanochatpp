cmake_minimum_required(VERSION 3.31)
project(nanochatpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CTest)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/eigen/CMakeLists.txt")
    message(FATAL_ERROR "Eigen is missing. Run: git submodule update --init --recursive")
endif()
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
add_subdirectory(external/eigen EXCLUDE_FROM_ALL SYSTEM)

option(NANOCHAT_ENABLE_VALIDATION "Enable Tensor and TensorView runtime validation" ON)
add_library(nanochatpp_tensor INTERFACE)
target_link_libraries(nanochatpp_tensor INTERFACE Eigen3::Eigen)
target_include_directories(nanochatpp_tensor INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/headers)
target_compile_definitions(nanochatpp_tensor INTERFACE
    NANOCHAT_ENABLE_VALIDATION=$<BOOL:${NANOCHAT_ENABLE_VALIDATION}>)

add_executable(nanochatpp main.cpp)
target_link_libraries(nanochatpp PRIVATE nanochatpp_tensor)

if(BUILD_TESTING)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.18.0
        GIT_SHALLOW TRUE
        SYSTEM
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    add_subdirectory(tests)
endif()
