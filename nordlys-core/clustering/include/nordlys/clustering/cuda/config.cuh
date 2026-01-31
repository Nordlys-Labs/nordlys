#pragma once
#ifdef NORDLYS_HAS_CUDA

#include <cuda_runtime.h>
#include <concepts>
#include <unordered_map>

namespace nordlys::clustering::cuda {

struct KernelConfig {
  int block_size{128};
  int grid_size{1};
  size_t shared_mem{0};
};

template <typename Kernel>
concept CudaKernel = std::is_pointer_v<Kernel>;

template <CudaKernel Kernel>
[[nodiscard]] auto get_optimal_config(Kernel kernel, int n_elements, size_t dynamic_smem = 0) 
  noexcept -> KernelConfig 
{
  KernelConfig config;
  int min_grid_size;
  
  cudaOccupancyMaxPotentialBlockSize(
    &min_grid_size,
    &config.block_size,
    kernel,
    dynamic_smem,
    0
  );
  
  config.grid_size = (n_elements + config.block_size - 1) / config.block_size;
  config.grid_size = std::max(config.grid_size, min_grid_size);
  config.shared_mem = dynamic_smem;
  
  return config;
}

} // namespace nordlys::clustering::cuda

#endif
