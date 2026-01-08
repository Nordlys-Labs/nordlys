# =============================================================================
# Nordlys Core - Benchmarks
# =============================================================================

find_package(benchmark REQUIRED)

# Create profiling output directory at configure time (CMake best practice)
file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/profiling")

# Tracy profiler integration
if(NORDLYS_ENABLE_TRACY)
  find_package(Tracy REQUIRED)
endif()

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
    $<$<BOOL:${NORDLYS_ENABLE_TRACY}>:Tracy::TracyClient>
)

target_include_directories(bench_nordlys_core PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Tracy profiling support
if(NORDLYS_ENABLE_TRACY)
  target_compile_definitions(bench_nordlys_core PRIVATE TRACY_ENABLE)
  message(STATUS "Tracy profiler enabled for benchmarks")
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
      $<$<BOOL:${NORDLYS_ENABLE_TRACY}>:Tracy::TracyClient>
  )
  
  target_include_directories(bench_nordlys_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  
  # Tracy profiling support for CUDA benchmarks
  if(NORDLYS_ENABLE_TRACY)
    target_compile_definitions(bench_nordlys_cuda PRIVATE TRACY_ENABLE)
  endif()
  
  message(STATUS "CUDA benchmarks enabled (bench_nordlys_cuda)")
endif()

# Tracy convenience target
if(NORDLYS_ENABLE_TRACY)
  set(PROFILE_SCRIPTS "${CMAKE_CURRENT_SOURCE_DIR}/scripts")
  
  add_custom_target(bench_tracy
    COMMAND "${PROFILE_SCRIPTS}/run_tracy.sh" "RoutingSingle_Medium"
    DEPENDS bench_nordlys_core
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Running Tracy profiler on RoutingSingle_Medium"
  )
  
  message(STATUS "Tracy targets available:")
  message(STATUS "  - bench_tracy: Quick Tracy profiling session")
endif()
