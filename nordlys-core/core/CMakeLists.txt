# =============================================================================
# Nordlys Core - C++ Library
# =============================================================================

include(FetchContent)

# SimSIMD (required by USearch for SIMD-accelerated distance computations)
FetchContent_Declare(
  simsimd
  GIT_REPOSITORY https://github.com/ashvardanian/SimSIMD.git
  GIT_TAG        v6.5.12
  GIT_SHALLOW    TRUE
  SYSTEM
)
set(SIMSIMD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SIMSIMD_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(simsimd)

# USearch (CPU vector search with SimSIMD backend)
FetchContent_Declare(
  usearch
  GIT_REPOSITORY https://github.com/unum-cloud/usearch.git
  GIT_TAG        v2.16.8
  GIT_SHALLOW    TRUE
  SYSTEM
)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)
set(USEARCH_BUILD_TEST OFF CACHE BOOL "" FORCE)
set(USEARCH_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(USEARCH_USE_OPENMP OFF CACHE BOOL "" FORCE)
set(USEARCH_USE_FP16LIB OFF CACHE BOOL "" FORCE)
set(USEARCH_USE_SIMSIMD ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(usearch)

# Core library
add_library(nordlys_core STATIC
  src/cluster_backend.cpp
  src/scorer.cpp
  src/checkpoint.cpp
)
add_library(nordlys::core ALIAS nordlys_core)

target_include_directories(nordlys_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  # SimSIMD headers needed by public usearch API
  SYSTEM PUBLIC
    $<BUILD_INTERFACE:${simsimd_SOURCE_DIR}/include>
)

# Common dependencies (always required)
target_link_libraries(nordlys_core
  PUBLIC
    nlohmann_json::nlohmann_json
    msgpack-cxx
    # USearch is in public API (CpuClusterBackendT uses it)
    # This is conditional via #ifndef __CUDACC__ so CUDA files won't see it
    usearch
)

# Tracy profiler (optional)
# Must be PUBLIC because tracy.hpp is a public header - consumers need TRACY_ENABLE
# defined to know whether to include <tracy/Tracy.hpp>
if(NORDLYS_ENABLE_TRACY)
  target_link_libraries(nordlys_core PUBLIC Tracy::TracyClient)
  target_compile_definitions(nordlys_core PUBLIC TRACY_ENABLE)
endif()

# Apply common compile options
nordlys_target_compile_options(nordlys_core)

# Suppress MSVC unreachable code warning in SimSIMD headers (external dependency)
# Even though SimSIMD is marked as SYSTEM, MSVC still treats warnings as errors
if(MSVC)
  target_compile_options(nordlys_core PRIVATE /wd4702)
endif()

# CUDA acceleration (optional)
# The CUDA backend is decoupled - it doesn't compile with usearch headers
# due to #ifndef __CUDACC__ guards in cluster_backend.hpp
if(NORDLYS_ENABLE_CUDA)
  add_subdirectory(src/cuda)
  target_link_libraries(nordlys_core PUBLIC nordlys::cuda)
  target_compile_definitions(nordlys_core PUBLIC NORDLYS_HAS_CUDA=1)
endif()

# Tests
if(NORDLYS_BUILD_TESTS)
  add_subdirectory(tests)
endif()

# Installation
install(
  TARGETS nordlys_core
  EXPORT nordlys_coreTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(
  DIRECTORY include/nordlys_core
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)
