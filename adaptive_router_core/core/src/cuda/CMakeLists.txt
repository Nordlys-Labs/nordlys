# =============================================================================
# Adaptive Router Core - CUDA Backend
# =============================================================================

find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "80;86;90")
endif()

add_library(adaptive_cuda STATIC
  cluster_cuda.cu
)
add_library(adaptive::cuda ALIAS adaptive_cuda)

target_include_directories(adaptive_cuda
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
)

target_link_libraries(adaptive_cuda
  PRIVATE
    CUDA::cublas
    CUDA::cudart
    Eigen3::Eigen
)

set_target_properties(adaptive_cuda PROPERTIES
  CUDA_STANDARD 20
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)

target_compile_definitions(adaptive_cuda PRIVATE ADAPTIVE_HAS_CUDA=1)