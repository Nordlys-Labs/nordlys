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

# Clustering library
add_library(nordlys_clustering STATIC src/cluster_cpu.cpp src/cluster_factory.cpp)
add_library(Nordlys::Clustering ALIAS nordlys_clustering)

target_include_directories(nordlys_clustering
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_include_directories(nordlys_clustering SYSTEM PUBLIC
  $<BUILD_INTERFACE:${simsimd_SOURCE_DIR}/include>
)

target_link_libraries(nordlys_clustering
  PUBLIC
    Nordlys::Common
  PRIVATE
    $<BUILD_INTERFACE:usearch>
)

# CUDA acceleration (optional)
if(NORDLYS_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_sources(nordlys_clustering PRIVATE src/cluster_cuda.cu)
  target_compile_definitions(nordlys_clustering PUBLIC NORDLYS_HAS_CUDA=1)
  target_link_libraries(nordlys_clustering PUBLIC CUDA::cublas CUDA::cudart)
  set_target_properties(nordlys_clustering PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )
  set_source_files_properties(src/cluster_cuda.cu PROPERTIES LANGUAGE CUDA)
endif()

# OpenMP parallelization (optional)
if(APPLE)
  execute_process(
    COMMAND brew --prefix libomp
    OUTPUT_VARIABLE LIBOMP_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE BREW_RESULT
  )
  if(BREW_RESULT EQUAL 0 AND EXISTS "${LIBOMP_PREFIX}")
    target_compile_options(nordlys_clustering PUBLIC -Xpreprocessor -fopenmp)
    target_include_directories(nordlys_clustering SYSTEM PUBLIC "${LIBOMP_PREFIX}/include")
    target_link_libraries(nordlys_clustering PUBLIC "${LIBOMP_PREFIX}/lib/libomp.dylib")
    message(STATUS "OpenMP found via Homebrew: ${LIBOMP_PREFIX}")
  else()
    message(STATUS "OpenMP not found (install with: brew install libomp)")
  endif()
else()
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(nordlys_clustering PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
  endif()
endif()

nordlys_target_compile_options(nordlys_clustering)

# Suppress MSVC unreachable code warning in SimSIMD headers (external dependency)
if(MSVC)
  target_compile_options(nordlys_clustering PRIVATE /wd4702)
endif()

if(NORDLYS_BUILD_TESTS)
  add_subdirectory(tests)
endif()

install(TARGETS nordlys_clustering
  EXPORT NordlysTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/nordlys
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
