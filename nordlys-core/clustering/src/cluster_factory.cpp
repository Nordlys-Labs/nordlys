#include <nordlys/clustering/cluster.hpp>
#include <nordlys/clustering/cluster_cpu.hpp>

#ifdef NORDLYS_HAS_CUDA
#  include <nordlys/clustering/cluster_cuda.hpp>
#else
bool cuda_available() noexcept { return false; }
#endif

std::unique_ptr<IClusterBackend> create_backend(Device device) {
  return std::visit(overloaded{[](CpuDevice) -> std::unique_ptr<IClusterBackend> {
                                 return std::make_unique<CpuClusterBackend>();
                               },
                               [](CudaDevice) -> std::unique_ptr<IClusterBackend> {
#ifdef NORDLYS_HAS_CUDA
                                 if (cuda_available()) {
                                   return std::make_unique<CudaClusterBackend>();
                                 }
                                 throw std::runtime_error("CUDA not available");
#else
                                 throw std::runtime_error("CUDA backend not compiled");
#endif
                               }},
                    device);
}
