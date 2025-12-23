#include <nordlys_core/cluster_backend.hpp>

#ifdef NORDLYS_HAS_CUDA
#include <nordlys_core/cuda/cluster_cuda.hpp>
#include <cuda_runtime.h>
#endif

bool cuda_available() noexcept {
#ifdef NORDLYS_HAS_CUDA
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  return err == cudaSuccess && device_count > 0;
#else
  return false;
#endif
}

// Float specialization
template<>
std::unique_ptr<IClusterBackendT<float>> create_cluster_backend(ClusterBackendType type) {
  switch (type) {
    case ClusterBackendType::CPU:
      return std::make_unique<CpuClusterBackendT<float>>();

    case ClusterBackendType::CUDA:
#ifdef NORDLYS_HAS_CUDA
      if (cuda_available()) {
        return std::make_unique<CudaClusterBackendT<float>>();
      }
#endif
      return std::make_unique<CpuClusterBackendT<float>>();

    case ClusterBackendType::Auto:
    default:
#ifdef NORDLYS_HAS_CUDA
      if (cuda_available()) {
        return std::make_unique<CudaClusterBackendT<float>>();
      }
#endif
      return std::make_unique<CpuClusterBackendT<float>>();
  }
}

// Double specialization
template<>
std::unique_ptr<IClusterBackendT<double>> create_cluster_backend(ClusterBackendType type) {
  switch (type) {
    case ClusterBackendType::CPU:
      return std::make_unique<CpuClusterBackendT<double>>();

    case ClusterBackendType::CUDA:
#ifdef NORDLYS_HAS_CUDA
      if (cuda_available()) {
        return std::make_unique<CudaClusterBackendT<double>>();
      }
#endif
      return std::make_unique<CpuClusterBackendT<double>>();

    case ClusterBackendType::Auto:
    default:
#ifdef NORDLYS_HAS_CUDA
      if (cuda_available()) {
        return std::make_unique<CudaClusterBackendT<double>>();
      }
#endif
      return std::make_unique<CpuClusterBackendT<double>>();
  }
}
