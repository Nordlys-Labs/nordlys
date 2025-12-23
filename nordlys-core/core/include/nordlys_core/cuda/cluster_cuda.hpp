#pragma once

#ifdef NORDLYS_HAS_CUDA

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <utility>
#include <vector>

#include <nordlys_core/cluster_backend.hpp>

// Templated CUDA backend using cuBLAS for GPU-accelerated cluster assignment
// Uses L2 distance trick: ||a-b||² = ||a||² + ||b||² - 2(a·b)
template<typename Scalar>
class CudaClusterBackendT : public IClusterBackendT<Scalar> {
public:
  CudaClusterBackendT();
  ~CudaClusterBackendT() override;

  // Non-copyable, non-movable
  CudaClusterBackendT(const CudaClusterBackendT&) = delete;
  CudaClusterBackendT& operator=(const CudaClusterBackendT&) = delete;
  CudaClusterBackendT(CudaClusterBackendT&&) = delete;
  CudaClusterBackendT& operator=(CudaClusterBackendT&&) = delete;

  void load_centroids(const Scalar* data, int n_clusters, int dim) override;

  [[nodiscard]] std::pair<int, Scalar> assign(const Scalar* embedding, int dim) override;

  [[nodiscard]] int get_n_clusters() const noexcept override { return n_clusters_; }

  [[nodiscard]] int get_dim() const noexcept override { return dim_; }

  [[nodiscard]] bool is_gpu_accelerated() const noexcept override { return true; }

private:
  // cuBLAS handle
  cublasHandle_t cublas_handle_ = nullptr;

  // Device memory pointers
  Scalar* d_centroids_ = nullptr;       // [dim x n_clusters] column-major for cuBLAS
  Scalar* d_centroid_norms_ = nullptr;  // [n_clusters] precomputed ||c_i||²
  Scalar* d_embedding_ = nullptr;       // [dim]
  Scalar* d_dots_ = nullptr;            // [n_clusters] workspace for dot products
  int* d_best_idx_ = nullptr;           // [1] result: best cluster index
  Scalar* d_best_dist_ = nullptr;       // [1] result: best distance

  int n_clusters_ = 0;
  int dim_ = 0;

  // CUDA stream for async operations
  cudaStream_t stream_ = nullptr;

  void free_device_memory();
};

#endif  // NORDLYS_HAS_CUDA
