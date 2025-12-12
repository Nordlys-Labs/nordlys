# =============================================================================
# Adaptive Router Core - High-Performance Model Routing Library
# =============================================================================
cmake_minimum_required(VERSION 3.24)

# -----------------------------------------------------------------------------
# Read version from pyproject.toml (single source of truth)
# -----------------------------------------------------------------------------
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" PYPROJECT_CONTENT)
string(REGEX MATCH "version = \"([0-9]+\\.[0-9]+\\.[0-9]+)\"" _ ${PYPROJECT_CONTENT})
set(ADAPTIVE_VERSION ${CMAKE_MATCH_1})
if(NOT ADAPTIVE_VERSION)
    message(FATAL_ERROR "Could not extract version from pyproject.toml")
endif()
message(STATUS "[adaptive_router_core] Version: ${ADAPTIVE_VERSION}")

# -----------------------------------------------------------------------------
# Project-level options (must be before project())
# -----------------------------------------------------------------------------
option(ADAPTIVE_ENABLE_CUDA "Enable CUDA GPU acceleration (AUTO=auto-detect)" AUTO)
option(ADAPTIVE_BUILD_PYTHON "Build Python bindings" ON)
option(ADAPTIVE_BUILD_TESTS "Build test suite" OFF)
option(ADAPTIVE_ENABLE_SANITIZERS "Enable address and undefined behavior sanitizers" OFF)
option(ADAPTIVE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)

# -----------------------------------------------------------------------------
# CUDA auto-detection (before project() for proper language setup)
# -----------------------------------------------------------------------------
if(ADAPTIVE_ENABLE_CUDA STREQUAL "AUTO")
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    set(ADAPTIVE_ENABLE_CUDA ON)
    message(STATUS "[adaptive_router_core] CUDA auto-detected: ${CMAKE_CUDA_COMPILER}")
  else()
    set(ADAPTIVE_ENABLE_CUDA OFF)
    message(STATUS "[adaptive_router_core] CUDA not found - GPU acceleration disabled")
  endif()
endif()

# -----------------------------------------------------------------------------
# Project declaration
# -----------------------------------------------------------------------------
if(ADAPTIVE_ENABLE_CUDA)
  project(adaptive_router_core
    VERSION ${ADAPTIVE_VERSION}
    DESCRIPTION "High-performance cluster-based model routing library"
    HOMEPAGE_URL "https://github.com/adaptive/adaptive_router"
    LANGUAGES CXX CUDA
  )
else()
  project(adaptive_router_core
    VERSION ${ADAPTIVE_VERSION}
    DESCRIPTION "High-performance cluster-based model routing library"
    HOMEPAGE_URL "https://github.com/adaptive/adaptive_router"
    LANGUAGES CXX
  )
endif()

# -----------------------------------------------------------------------------
# Standard install directories (must be before any use of CMAKE_INSTALL_*)
# -----------------------------------------------------------------------------
include(GNUInstallDirs)

# -----------------------------------------------------------------------------
# Prevent in-source builds
# -----------------------------------------------------------------------------
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "In-source builds are not allowed. Use: cmake -B build")
endif()

# -----------------------------------------------------------------------------
# Global compiler settings
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(ADAPTIVE_ENABLE_CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
endif()

# Export compile commands for IDE/tooling support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Position independent code (required for shared libraries)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Symbol visibility (hide by default, export explicitly)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# -----------------------------------------------------------------------------
# Conan integration (find toolchain in standard locations)
# -----------------------------------------------------------------------------
set(_conan_toolchain_paths
  "${CMAKE_BINARY_DIR}/conan_toolchain.cmake"
  "${CMAKE_SOURCE_DIR}/build/build/Release/generators/conan_toolchain.cmake"
  "${CMAKE_SOURCE_DIR}/build/build/Debug/generators/conan_toolchain.cmake"
)

foreach(_toolchain IN LISTS _conan_toolchain_paths)
  if(EXISTS "${_toolchain}")
    message(STATUS "[adaptive_router_core] Using Conan toolchain: ${_toolchain}")
    include("${_toolchain}")
    get_filename_component(_generators_dir "${_toolchain}" DIRECTORY)
    list(APPEND CMAKE_PREFIX_PATH "${_generators_dir}")
    break()
  endif()
endforeach()
unset(_conan_toolchain_paths)
unset(_toolchain)
unset(_generators_dir)

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(msgpack-cxx REQUIRED)

# -----------------------------------------------------------------------------
# Helper function: Apply common compile options to a target
# -----------------------------------------------------------------------------
function(adaptive_target_compile_options target)
  # Warnings
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(_warn_base
      -Wall
      -Wextra
      -Wpedantic
      -Wconversion
      -Wsign-conversion
      -Wnon-virtual-dtor
      -Woverloaded-virtual
      -Wold-style-cast
      -Wcast-qual
      -Wformat=2
      -Wimplicit-fallthrough
      -Wmissing-include-dirs
    )
    set(_warn_base_cuda ${_warn_base})
    list(REMOVE_ITEM _warn_base_cuda -Wpedantic -Wold-style-cast)
    set(_warn_base_cuda_prefixed)
    foreach(_w IN LISTS _warn_base_cuda)
      list(APPEND _warn_base_cuda_prefixed "-Xcompiler=${_w}")
    endforeach()

    # Apply host warnings for both C++ and CUDA compilations
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:${_warn_base}>
      $<$<COMPILE_LANGUAGE:CUDA>:${_warn_base_cuda_prefixed}>
      $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:${ADAPTIVE_WARNINGS_AS_ERRORS}>>:-Werror>
      $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${ADAPTIVE_WARNINGS_AS_ERRORS}>>:-Xcompiler=-Werror>
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )

    # GCC-specific warnings
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set(_warn_gcc
        -Wlogical-op
        -Wduplicated-cond
        -Wduplicated-branches
      )
      set(_warn_gcc_cuda)
      foreach(_w IN LISTS _warn_gcc)
        list(APPEND _warn_gcc_cuda "-Xcompiler=${_w}")
      endforeach()
      target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${_warn_gcc}>
        $<$<COMPILE_LANGUAGE:CUDA>:${_warn_gcc_cuda}>
      )
    endif()

    # Clang-specific warnings
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wshadow-all>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wshadow-all>
      )
    endif()

  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${target} PRIVATE
      /W4
      /permissive-
      /Zc:__cplusplus
      /Zc:preprocessor
      $<$<BOOL:${ADAPTIVE_WARNINGS_AS_ERRORS}>:/WX>
    )
  endif()

  # Sanitizers (Debug builds only)
  if(ADAPTIVE_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE
      $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-fsanitize=address,undefined>
      $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-fno-omit-frame-pointer>
      $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-fno-sanitize-recover=all>
    )
    target_link_options(${target} PRIVATE
      $<$<CONFIG:Debug>:-fsanitize=address,undefined>
    )
  endif()
endfunction()

# -----------------------------------------------------------------------------
# Core library
# -----------------------------------------------------------------------------
add_library(adaptive_core STATIC)
add_library(adaptive::core ALIAS adaptive_core)

target_sources(adaptive_core PRIVATE
  src/router.cpp
  src/cluster_backend.cpp
  src/scorer.cpp
  src/profile.cpp
)

target_include_directories(adaptive_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(adaptive_core
  PUBLIC
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    msgpack-cxx
)

target_compile_features(adaptive_core PUBLIC cxx_std_20)
adaptive_target_compile_options(adaptive_core)

# -----------------------------------------------------------------------------
# CUDA acceleration (optional)
# -----------------------------------------------------------------------------
if(ADAPTIVE_ENABLE_CUDA)
  message(STATUS "[adaptive_router_core] CUDA GPU acceleration enabled")

  find_package(CUDAToolkit REQUIRED)

  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "80;86;90" CACHE STRING "CUDA architectures to build (semicolon-separated)" FORCE)
  endif()

  add_library(adaptive_core_cuda STATIC)
  add_library(adaptive::cuda ALIAS adaptive_core_cuda)

  target_sources(adaptive_core_cuda PRIVATE
    src/cuda/cluster_cuda.cu
  )

  target_include_directories(adaptive_core_cuda
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

  target_link_libraries(adaptive_core_cuda
    PRIVATE
      CUDA::cublas
      CUDA::cudart
      Eigen3::Eigen
  )

  set_target_properties(adaptive_core_cuda PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
  )

  # Define CUDA macro so the .cu file sees it during compilation
  target_compile_definitions(adaptive_core_cuda PRIVATE ADAPTIVE_HAS_CUDA=1)
  adaptive_target_compile_options(adaptive_core_cuda)

  # Link CUDA support into core library
  target_link_libraries(adaptive_core PUBLIC adaptive_core_cuda)
  target_compile_definitions(adaptive_core PUBLIC ADAPTIVE_HAS_CUDA=1)
  # Provide CUDA include paths to adaptive_core since it includes CUDA headers
  # Treat CUDA toolkit headers as system to avoid propagating their warnings
  target_include_directories(adaptive_core SYSTEM PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
else()
  message(STATUS "[adaptive_router_core] CUDA GPU acceleration disabled")
endif()

# -----------------------------------------------------------------------------
# C API (shared library)
# -----------------------------------------------------------------------------
add_library(adaptive_c SHARED)
add_library(adaptive::c ALIAS adaptive_c)

target_sources(adaptive_c PRIVATE
  bindings/c/adaptive_c.cpp
)

target_include_directories(adaptive_c
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/bindings/c>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(adaptive_c PRIVATE adaptive_core)
target_compile_features(adaptive_c PRIVATE cxx_std_20)
target_compile_definitions(adaptive_c PRIVATE ADAPTIVE_C_EXPORTS)
adaptive_target_compile_options(adaptive_c)

set_target_properties(adaptive_c PROPERTIES
  OUTPUT_NAME "adaptive_c"
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
)

if(WIN32)
  set_target_properties(adaptive_c PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS OFF  # Use explicit ADAPTIVE_API macros
  )
endif()

# -----------------------------------------------------------------------------
# Python bindings (optional)
# -----------------------------------------------------------------------------
if(ADAPTIVE_BUILD_PYTHON)
  add_subdirectory(bindings/python)
endif()

# -----------------------------------------------------------------------------
# Tests (optional)
# -----------------------------------------------------------------------------
if(ADAPTIVE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# -----------------------------------------------------------------------------
# Development tools
# -----------------------------------------------------------------------------

# clang-format target
find_program(CLANG_FORMAT NAMES clang-format-18 clang-format-17 clang-format)
if(CLANG_FORMAT)
  file(GLOB_RECURSE _format_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.cuh"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/bindings/**/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/bindings/**/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/bindings/**/*.h"
  )
  add_custom_target(format
    COMMAND ${CLANG_FORMAT} -i ${_format_sources}
    COMMENT "Running clang-format"
    VERBATIM
  )
  add_custom_target(format-check
    COMMAND ${CLANG_FORMAT} --dry-run --Werror ${_format_sources}
    COMMENT "Checking clang-format compliance"
    VERBATIM
  )
  unset(_format_sources)
endif()

# clang-tidy target
find_program(CLANG_TIDY NAMES clang-tidy-18 clang-tidy-17 clang-tidy)
if(CLANG_TIDY AND EXISTS "${CMAKE_BINARY_DIR}/compile_commands.json")
  file(GLOB_RECURSE _tidy_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/bindings/**/*.cpp"
  )
  add_custom_target(tidy
    COMMAND ${CLANG_TIDY} -p "${CMAKE_BINARY_DIR}" ${_tidy_sources}
    COMMENT "Running clang-tidy"
    VERBATIM
  )
  unset(_tidy_sources)
endif()

# -----------------------------------------------------------------------------
# Installation (CMake package config)
# -----------------------------------------------------------------------------
include(CMakePackageConfigHelpers)

# Build list of targets to install
set(_install_targets adaptive_core adaptive_c)
if(ADAPTIVE_ENABLE_CUDA)
  list(APPEND _install_targets adaptive_core_cuda)
endif()

install(TARGETS ${_install_targets}
  EXPORT adaptive_router_coreTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT adaptive_router_coreTargets
  FILE adaptive_router_coreTargets.cmake
  NAMESPACE adaptive::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/adaptive_router_core
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/adaptive_router_coreConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/adaptive_router_coreConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/adaptive_router_core
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/adaptive_router_coreConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/adaptive_router_coreConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/adaptive_router_core
)

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "=== adaptive_router_core ${PROJECT_VERSION} ===")
message(STATUS "  C++ Standard:     ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA:             ${ADAPTIVE_ENABLE_CUDA}")
message(STATUS "  Python bindings:  ${ADAPTIVE_BUILD_PYTHON}")
message(STATUS "  Tests:            ${ADAPTIVE_BUILD_TESTS}")
message(STATUS "  Sanitizers:       ${ADAPTIVE_ENABLE_SANITIZERS}")
message(STATUS "  Warnings as errors: ${ADAPTIVE_WARNINGS_AS_ERRORS}")
message(STATUS "")
