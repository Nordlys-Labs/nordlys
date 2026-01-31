#pragma once

#include <nordlys/clustering/cluster.hpp>

#ifdef NORDLYS_HAS_CUDA

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <nordlys/clustering/cuda/memory.cuh>
#include <nordlys/clustering/cuda/memory_pool.cuh>

namespace nordlys::clustering {

class CudaClusterBackend : public IClusterBackend {
public:
  CudaClusterBackend();
  ~CudaClusterBackend() override;

  CudaClusterBackend(const CudaClusterBackend&) = delete;
  CudaClusterBackend& operator=(const CudaClusterBackend&) = delete;
  CudaClusterBackend(CudaClusterBackend&&) = delete;
  CudaClusterBackend& operator=(CudaClusterBackend&&) = delete;

  void load_centroids(const float* data, size_t n_clusters, size_t dim) override;

  [[nodiscard]] auto assign(EmbeddingView view) -> std::pair<int, float> override;

  [[nodiscard]] auto assign_batch(EmbeddingBatchView view)
    -> std::vector<std::pair<int, float>> override;

  [[nodiscard]] auto n_clusters() const noexcept -> size_t override {
    return static_cast<size_t>(n_clusters_);
  }
  [[nodiscard]] auto dim() const noexcept -> size_t override { 
    return static_cast<size_t>(dim_); 
  }

private:
  void free_memory();
  void capture_graph();

  auto assign_batch_from_host(EmbeddingBatchView view)
    -> std::vector<std::pair<int, float>>;
  auto assign_batch_from_device(EmbeddingBatchView view)
    -> std::vector<std::pair<int, float>>;

  cublasHandle_t cublas_ = nullptr;
  cudaStream_t stream_ = nullptr;
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t graph_exec_ = nullptr;
  bool graph_valid_ = false;

  std::unique_ptr<cuda::MemoryPool> memory_pool_;

  cuda::DevicePtr<float> d_centroids_;       // Col-major for cuBLAS (batch path)
  cuda::DevicePtr<float> d_centroids_row_;   // Row-major for single-query path (coalesced access)
  cuda::DevicePtr<float> d_centroid_norms_;
  cuda::DevicePtr<float> d_embedding_;
  cuda::DevicePtr<float> d_embed_norm_;
  cuda::DevicePtr<float> d_dots_;
  cuda::DevicePtr<int> d_best_idx_;
  cuda::DevicePtr<float> d_best_dist_;

  cuda::PinnedPtr<float> h_embedding_;
  cuda::PinnedPtr<int> h_best_idx_;
  cuda::PinnedPtr<float> h_best_dist_;

  static constexpr int kNumPipelineStages = 4;

  struct PipelineStage {
    cudaStream_t stream = nullptr;
    cudaEvent_t event = nullptr;
    cuda::DevicePtr<float> d_queries;
    cuda::DevicePtr<float> d_norms;
    cuda::DevicePtr<float> d_dots;
    cuda::DevicePtr<char> d_results;   // Packed ClusterResult<float> array
    cuda::PinnedPtr<float> h_queries;
    cuda::PinnedPtr<char> h_results;   // Packed ClusterResult<float> array
    int capacity = 0;
  };

  PipelineStage stages_[kNumPipelineStages];
  cublasHandle_t pipeline_cublas_[kNumPipelineStages] = {};
  bool pipeline_initialized_ = false;

  void init_pipeline();
  void ensure_stage_capacity(int stage_idx, int count);

  int n_clusters_ = 0;
  int dim_ = 0;
};

} // namespace nordlys::clustering

#endif  // NORDLYS_HAS_CUDA
