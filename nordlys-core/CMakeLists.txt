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
option(NORDLYS_ENABLE_TRACY "Enable Tracy profiler integration" OFF)

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

# =============================================================================
# Common Dependencies (Conan-managed)
# =============================================================================
find_package(nlohmann_json REQUIRED)
find_package(msgpack-cxx REQUIRED)

# Tracy profiler (optional)
# NOTE: Do NOT use add_compile_definitions(TRACY_ENABLE) here - it would affect
# all targets globally, but only targets that link Tracy::TracyClient have access
# to the headers. Each target that wants Tracy must add the definition itself.
if(NORDLYS_ENABLE_TRACY)
  find_package(Tracy REQUIRED)
  message(STATUS "Tracy profiler enabled")
endif()

# Enable testing
if(NORDLYS_BUILD_TESTS)
  include(CTest)
  enable_testing()
endif()

# Add modules
add_subdirectory(core)
add_subdirectory(bindings)

# Benchmarks
if(NORDLYS_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# Summary
message(STATUS "")
message(STATUS "=== nordlys_core ${PROJECT_VERSION} ===")
message(STATUS "  CUDA:         ${NORDLYS_ENABLE_CUDA}")
message(STATUS "  Python:       ${NORDLYS_BUILD_PYTHON}")
message(STATUS "  C bindings:   ${NORDLYS_BUILD_C}")
message(STATUS "  Tests:        ${NORDLYS_BUILD_TESTS}")
message(STATUS "  Benchmarks:   ${NORDLYS_BUILD_BENCHMARKS}")
message(STATUS "  Tracy:        ${NORDLYS_ENABLE_TRACY}")
message(STATUS "")
