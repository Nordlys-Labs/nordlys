# =============================================================================
# Nordlys Core - CUDA Backend
# =============================================================================

find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "80;86;90")
endif()

add_library(nordlys_cuda STATIC
  cluster_cuda.cu
)
add_library(nordlys::cuda ALIAS nordlys_cuda)

target_include_directories(nordlys_cuda
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
)

target_link_libraries(nordlys_cuda
  PUBLIC
    CUDA::cublas
    CUDA::cudart
)

target_compile_definitions(nordlys_cuda 
  PRIVATE 
    NORDLYS_HAS_CUDA=1
)

set_target_properties(nordlys_cuda PROPERTIES
  CUDA_STANDARD 20
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)

target_compile_options(nordlys_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
    -Xcudafe --diag_suppress=20015
  >
)
