# =============================================================================
# Nordlys Core - Benchmarks
# =============================================================================

find_package(benchmark REQUIRED)

# Create profiling output directory at configure time (CMake best practice)
file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/profiling")

# CPU Benchmarks (always built when NORDLYS_BUILD_BENCHMARKS=ON)
add_executable(bench_nordlys_core
  bench_routing_e2e.cpp
  bench_checkpoint_e2e.cpp
)

target_link_libraries(bench_nordlys_core
  PRIVATE
    nordlys::core
    benchmark::benchmark
    benchmark::benchmark_main
)

target_include_directories(bench_nordlys_core PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Profiling support (add debug symbols and frame pointers)
if(NORDLYS_BUILD_PROFILE)
  target_compile_options(bench_nordlys_core PRIVATE
    -g
    -fno-omit-frame-pointer
  )
  message(STATUS "Profiling symbols enabled for benchmarks")
endif()

# Copy fixtures to build directory at build-time
file(GLOB FIXTURE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/*.json)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/fixtures/.fixtures_copied
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${CMAKE_CURRENT_SOURCE_DIR}/fixtures
          ${CMAKE_CURRENT_BINARY_DIR}/fixtures
  COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/fixtures/.fixtures_copied
  DEPENDS ${FIXTURE_FILES}
  COMMENT "Copying benchmark fixtures to build directory"
)

add_custom_target(copy_benchmark_fixtures
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/fixtures/.fixtures_copied
)

add_dependencies(bench_nordlys_core copy_benchmark_fixtures)

# CUDA Benchmarks (conditional, only if CUDA enabled)
if(NORDLYS_ENABLE_CUDA)
  add_executable(bench_nordlys_cuda
    bench_routing_cuda.cpp
  )
  
  target_link_libraries(bench_nordlys_cuda
    PRIVATE
      nordlys::core
      benchmark::benchmark
      benchmark::benchmark_main
  )
  
  target_include_directories(bench_nordlys_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  
  # Profiling support for CUDA benchmarks
  if(NORDLYS_BUILD_PROFILE)
    target_compile_options(bench_nordlys_cuda PRIVATE
      -g
      -fno-omit-frame-pointer
    )
  endif()
  
  message(STATUS "CUDA benchmarks enabled (bench_nordlys_cuda)")
endif()

# Profiling convenience targets (require benchmarks to be built)
if(NORDLYS_BUILD_PROFILE)
  set(BENCH_BINARY "${CMAKE_CURRENT_BINARY_DIR}/bench_nordlys_core")
  set(PROFILE_SCRIPTS "${CMAKE_CURRENT_SOURCE_DIR}/scripts")
  
  add_custom_target(bench_profile
    COMMAND "${PROFILE_SCRIPTS}/profile_perf.sh" "RoutingSingle_Medium"
    DEPENDS bench_nordlys_core
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Running CPU performance profiling on RoutingSingle_Medium"
  )
  
  add_custom_target(bench_flamegraph
    COMMAND "${PROFILE_SCRIPTS}/profile_flamegraph.sh" "RoutingSingle_Medium"
    DEPENDS bench_nordlys_core
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Generating flamegraph for RoutingSingle_Medium"
  )
  
  add_custom_target(bench_profile_all
    COMMAND "${PROFILE_SCRIPTS}/profile_all.sh" "RoutingSingle_Medium"
    DEPENDS bench_nordlys_core
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Running comprehensive profiling suite"
  )
endif()
