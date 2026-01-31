# =============================================================================
# Nordlys Core - Benchmarks
# =============================================================================

find_package(benchmark REQUIRED)

# Create profiling output directory at configure time (CMake best practice)
file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/profiling")

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

# =============================================================================
# CPU Cluster Benchmarks
# =============================================================================

add_executable(bench_cluster_cpu
  bench_cluster_cpu.cpp
)

target_link_libraries(bench_cluster_cpu
  PRIVATE
    nordlys_clustering
    benchmark::benchmark
    benchmark::benchmark_main
)

target_include_directories(bench_cluster_cpu PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_dependencies(bench_cluster_cpu nordlys_clustering)

message(STATUS "CPU cluster benchmarks enabled (bench_cluster_cpu)")

# =============================================================================
# End-to-End Benchmarks (Checkpoint + Routing)
# =============================================================================

add_executable(bench_e2e
  bench_e2e.cpp
  bench_checkpoint_e2e.cpp
)

target_link_libraries(bench_e2e
  PRIVATE
    nordlys_core
    benchmark::benchmark
    benchmark::benchmark_main
)

target_include_directories(bench_e2e PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_dependencies(bench_e2e copy_benchmark_fixtures)

message(STATUS "End-to-end benchmarks enabled (bench_e2e)")

# =============================================================================
# CUDA Cluster Benchmarks (conditional)
# =============================================================================

if(NORDLYS_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  
  add_executable(bench_cluster_cuda
    bench_cluster_cuda.cu
  )
  
  set_target_properties(bench_cluster_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 20
  )
  
  target_link_libraries(bench_cluster_cuda
    PRIVATE
      nordlys_clustering
      benchmark::benchmark
      benchmark::benchmark_main
      CUDA::cudart
  )
  
  target_include_directories(bench_cluster_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  
  add_dependencies(bench_cluster_cuda nordlys_clustering)
  
  message(STATUS "CUDA cluster benchmarks enabled (bench_cluster_cuda)")
endif()
