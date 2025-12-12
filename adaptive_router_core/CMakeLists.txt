# =============================================================================
# Adaptive Router Core - Top-Level CMake
# =============================================================================
cmake_minimum_required(VERSION 3.24)

# Read version from dedicated VERSION file (decoupled from Python packaging)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" ADAPTIVE_VERSION)
string(STRIP "${ADAPTIVE_VERSION}" ADAPTIVE_VERSION)

# Options
option(ADAPTIVE_ENABLE_CUDA "Enable CUDA GPU acceleration (AUTO/ON/OFF)" AUTO)
option(ADAPTIVE_BUILD_TESTS "Build test suites" OFF)
option(ADAPTIVE_BUILD_PYTHON "Build Python bindings" ON)
option(ADAPTIVE_BUILD_C "Build C FFI bindings" ON)
option(ADAPTIVE_ENABLE_SANITIZERS "Enable sanitizers in Debug" OFF)
option(ADAPTIVE_WARNINGS_AS_ERRORS "Treat warnings as errors" ON)

# CUDA auto-detection
if(ADAPTIVE_ENABLE_CUDA STREQUAL "AUTO")
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    set(ADAPTIVE_ENABLE_CUDA ON)
  else()
    set(ADAPTIVE_ENABLE_CUDA OFF)
  endif()
endif()

# Project declaration
if(ADAPTIVE_ENABLE_CUDA)
  project(adaptive_router_core VERSION ${ADAPTIVE_VERSION} LANGUAGES CXX CUDA)
else()
  project(adaptive_router_core VERSION ${ADAPTIVE_VERSION} LANGUAGES CXX)
endif()

# Standard settings
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Include shared CMake utilities
include(cmake/AdaptiveCompileOptions.cmake)

# Dependencies
find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(msgpack-cxx REQUIRED)

# Enable testing
if(ADAPTIVE_BUILD_TESTS)
  include(CTest)
  enable_testing()
endif()

# Add modules
add_subdirectory(core)
add_subdirectory(bindings)

# Summary
message(STATUS "")
message(STATUS "=== adaptive_router_core ${PROJECT_VERSION} ===")
message(STATUS "  CUDA:       ${ADAPTIVE_ENABLE_CUDA}")
message(STATUS "  Python:     ${ADAPTIVE_BUILD_PYTHON}")
message(STATUS "  C bindings: ${ADAPTIVE_BUILD_C}")
message(STATUS "  Tests:      ${ADAPTIVE_BUILD_TESTS}")
message(STATUS "")