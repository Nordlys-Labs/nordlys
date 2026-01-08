#pragma once

#ifdef NORDLYS_HAS_CUDA

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <utility>

#include <nordlys_core/cluster_backend.hpp>

// CUDA backend for GPU-accelerated cluster assignment
// Uses L2 distance: ||a-b||² = ||a||² + ||b||² - 2(a·b)
// Implementation: cuBLAS GEMV + fused argmin + CUDA graphs + pinned memory
template<typename Scalar>
class CudaClusterBackendT : public IClusterBackendT<Scalar> {
public:
  CudaClusterBackendT();
  ~CudaClusterBackendT() override;

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
  cublasHandle_t cublas_ = nullptr;
  cudaStream_t stream_ = nullptr;
  
  // CUDA graph for near-zero launch overhead
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t graph_exec_ = nullptr;
  bool graph_valid_ = false;
  
  // Device memory (persistent)
  Scalar* d_centroids_ = nullptr;      // [dim x n_clusters] col-major
  Scalar* d_centroid_norms_ = nullptr; // [n_clusters]
  Scalar* d_embedding_ = nullptr;      // [dim]
  Scalar* d_embed_norm_ = nullptr;     // [1] for graph capture
  Scalar* d_dots_ = nullptr;           // [n_clusters]
  int* d_best_idx_ = nullptr;          // [1]
  Scalar* d_best_dist_ = nullptr;      // [1]
  
  // Pinned host memory
  Scalar* h_embedding_ = nullptr;      // [dim]
  int* h_best_idx_ = nullptr;          // [1]
  Scalar* h_best_dist_ = nullptr;      // [1]

  void free_memory();
  void capture_graph();

  int n_clusters_ = 0;
  int dim_ = 0;
};

#endif  // NORDLYS_HAS_CUDA
