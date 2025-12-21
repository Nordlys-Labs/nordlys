# =============================================================================
# Nordlys Core - Top-Level CMake
# =============================================================================
cmake_minimum_required(VERSION 3.24)

# Read version from dedicated VERSION file (decoupled from Python packaging)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" NORDLYS_VERSION)
string(STRIP "${NORDLYS_VERSION}" NORDLYS_VERSION)

# Options
option(NORDLYS_ENABLE_CUDA "Enable CUDA GPU acceleration (AUTO/ON/OFF)" AUTO)
option(NORDLYS_BUILD_TESTS "Build test suites" OFF)
option(NORDLYS_BUILD_PYTHON "Build Python bindings" ON)
option(NORDLYS_BUILD_C "Build C FFI bindings" ON)
option(NORDLYS_ENABLE_SANITIZERS "Enable sanitizers in Debug" OFF)
option(NORDLYS_WARNINGS_AS_ERRORS "Treat warnings as errors" ON)

# CUDA auto-detection
if(NORDLYS_ENABLE_CUDA STREQUAL "AUTO")
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    set(NORDLYS_ENABLE_CUDA ON)
  else()
    set(NORDLYS_ENABLE_CUDA OFF)
  endif()
endif()

# Project declaration
if(NORDLYS_ENABLE_CUDA)
  project(nordlys_core VERSION ${NORDLYS_VERSION} LANGUAGES CXX CUDA)
else()
  project(nordlys_core VERSION ${NORDLYS_VERSION} LANGUAGES CXX)
endif()

# Standard settings
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Include shared CMake utilities
include(cmake/NordlysCompileOptions.cmake)

# Dependencies
find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(msgpack-cxx REQUIRED)

# Enable testing
if(NORDLYS_BUILD_TESTS)
  include(CTest)
  enable_testing()
endif()

# Add modules
add_subdirectory(core)
add_subdirectory(bindings)

# Summary
message(STATUS "")
message(STATUS "=== nordlys_core ${PROJECT_VERSION} ===")
message(STATUS "  CUDA:       ${NORDLYS_ENABLE_CUDA}")
message(STATUS "  Python:     ${NORDLYS_BUILD_PYTHON}")
message(STATUS "  C bindings: ${NORDLYS_BUILD_C}")
message(STATUS "  Tests:      ${NORDLYS_BUILD_TESTS}")
message(STATUS "")