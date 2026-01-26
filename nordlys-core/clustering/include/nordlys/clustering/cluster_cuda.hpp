#pragma once

#include <nordlys/clustering/cluster.hpp>

#ifdef NORDLYS_HAS_CUDA

#  include <cublas_v2.h>
#  include <cuda_runtime.h>

#  include <nordlys/clustering/cuda/memory.cuh>

class CudaClusterBackend : public IClusterBackend {
public:
  CudaClusterBackend();
  ~CudaClusterBackend() override;

  CudaClusterBackend(const CudaClusterBackend&) = delete;
  CudaClusterBackend& operator=(const CudaClusterBackend&) = delete;
  CudaClusterBackend(CudaClusterBackend&&) = delete;
  CudaClusterBackend& operator=(CudaClusterBackend&&) = delete;

  void load_centroids(const float* data, size_t n_clusters, size_t dim) override;

  [[nodiscard]] std::pair<int, float> assign(EmbeddingView view) override;

  [[nodiscard]] std::vector<std::pair<int, float>> assign_batch(EmbeddingBatchView view) override;

  [[nodiscard]] size_t n_clusters() const noexcept override {
    return static_cast<size_t>(n_clusters_);
  }
  [[nodiscard]] size_t dim() const noexcept override { return static_cast<size_t>(dim_); }

private:
  void free_memory();
  void capture_graph();

  std::vector<std::pair<int, float>> assign_batch_from_host(EmbeddingBatchView view);
  std::vector<std::pair<int, float>> assign_batch_from_device(EmbeddingBatchView view);

  cublasHandle_t cublas_ = nullptr;
  cudaStream_t stream_ = nullptr;
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t graph_exec_ = nullptr;
  bool graph_valid_ = false;

  CudaDevicePtr<float> d_centroids_;
  CudaDevicePtr<float> d_centroid_norms_;
  CudaDevicePtr<float> d_embedding_;
  CudaDevicePtr<float> d_embed_norm_;
  CudaDevicePtr<float> d_dots_;
  CudaDevicePtr<int> d_best_idx_;
  CudaDevicePtr<float> d_best_dist_;

  CudaPinnedPtr<float> h_embedding_;
  CudaPinnedPtr<int> h_best_idx_;
  CudaPinnedPtr<float> h_best_dist_;

  static constexpr int kNumPipelineStages = 2;

  struct PipelineStage {
    cudaStream_t stream = nullptr;
    cudaEvent_t event = nullptr;
    CudaDevicePtr<float> d_queries;
    CudaDevicePtr<float> d_norms;
    CudaDevicePtr<float> d_dots;
    CudaDevicePtr<int> d_idx;
    CudaDevicePtr<float> d_dist;
    CudaPinnedPtr<float> h_queries;
    CudaPinnedPtr<int> h_idx;
    CudaPinnedPtr<float> h_dist;
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

#endif  // NORDLYS_HAS_CUDA
