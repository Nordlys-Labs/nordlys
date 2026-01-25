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
  bench_cluster_backend.cpp
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
  find_package(CUDAToolkit REQUIRED)
  
  add_executable(bench_nordlys_cuda
    bench_routing_cuda.cpp
  )
  
  target_link_libraries(bench_nordlys_cuda
    PRIVATE
      nordlys::core
      benchmark::benchmark
      benchmark::benchmark_main
      CUDA::cudart
      stdc++
  )
  
  target_include_directories(bench_nordlys_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  
  
  message(STATUS "CUDA benchmarks enabled (bench_nordlys_cuda)")
endif()

