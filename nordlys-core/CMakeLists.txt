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
option(NORDLYS_BUILD_BENCHMARKS "Build benchmark suite" OFF)

# CUDA auto-detection and configuration
if(NORDLYS_ENABLE_CUDA STREQUAL "AUTO")
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    set(NORDLYS_ENABLE_CUDA ON)
  else()
    set(NORDLYS_ENABLE_CUDA OFF)
  endif()
elseif(NORDLYS_ENABLE_CUDA)
  # When CUDA is explicitly enabled, set CMAKE_CUDA_COMPILER if not already set
  # This handles environment variables like CUDACXX set by CI/build systems
  if(NOT CMAKE_CUDA_COMPILER AND DEFINED ENV{CUDACXX})
    set(CMAKE_CUDA_COMPILER $ENV{CUDACXX})
  endif()
endif()

# Project declaration
if(NORDLYS_ENABLE_CUDA)
  # Use explicit CUDA architectures for cross-platform CI builds (GPU-less runners)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75;80;86;89;90)
  endif()
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
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# =============================================================================
# Common Dependencies (Conan-managed)
# =============================================================================
find_package(nlohmann_json REQUIRED)
find_package(msgpack-cxx REQUIRED)
find_package(simdjson REQUIRED)

# Enable testing
if(NORDLYS_BUILD_TESTS)
  include(CTest)
  enable_testing()
endif()

# =============================================================================
# Add Modules (in dependency order)
# =============================================================================
add_subdirectory(common)
add_subdirectory(scoring)
add_subdirectory(checkpoint)
add_subdirectory(clustering)
add_subdirectory(routing)

# =============================================================================
# Unified Interface Library
# =============================================================================
add_library(nordlys_core INTERFACE)
add_library(Nordlys::Core ALIAS nordlys_core)

target_link_libraries(nordlys_core INTERFACE
  Nordlys::Routing
  Nordlys::Clustering
  Nordlys::Scoring
  Nordlys::Checkpoint
  Nordlys::Common
)

# =============================================================================
# Integration Tests
# =============================================================================
if(NORDLYS_BUILD_TESTS)
  add_subdirectory(test)
endif()

# =============================================================================
# Bindings
# =============================================================================
add_subdirectory(bindings)

# =============================================================================
# Benchmarks
# =============================================================================
if(NORDLYS_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# =============================================================================
# Installation and Export (for find_package support)
# =============================================================================
install(TARGETS nordlys_core
  EXPORT NordlysTargets
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT NordlysTargets
  FILE NordlysTargets.cmake
  NAMESPACE Nordlys::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Nordlys
)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/NordlysConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/NordlysConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Nordlys
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/NordlysConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/NordlysConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/NordlysConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Nordlys
)

# Summary
message(STATUS "")
message(STATUS "=== nordlys_core ${PROJECT_VERSION} ===")
message(STATUS "  CUDA:         ${NORDLYS_ENABLE_CUDA}")
message(STATUS "  Python:       ${NORDLYS_BUILD_PYTHON}")
message(STATUS "  C bindings:   ${NORDLYS_BUILD_C}")
message(STATUS "  Tests:        ${NORDLYS_BUILD_TESTS}")
message(STATUS "  Benchmarks:   ${NORDLYS_BUILD_BENCHMARKS}")
message(STATUS "")
