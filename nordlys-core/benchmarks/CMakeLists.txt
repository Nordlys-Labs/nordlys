# =============================================================================
# Nordlys Core - Benchmarks
# =============================================================================

find_package(benchmark REQUIRED)

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

# Copy fixtures to build directory for easy access
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/fixtures
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

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
  
  message(STATUS "CUDA benchmarks enabled (bench_nordlys_cuda)")
endif()
