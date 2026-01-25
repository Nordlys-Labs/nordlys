#ifdef NORDLYS_HAS_CUDA

#  include <cmath>
#  include <cstring>
#  include <nordlys_core/cluster.hpp>
#  include <nordlys_core/cuda_common.cuh>
#  include <nordlys_core/cuda_distance.cuh>
#  include <numeric>
#  include <stdexcept>
#  include <string>
#  include <vector>

namespace {

  template <typename T>
  std::vector<T> compute_squared_norms_cpu(const T* data, size_t n, size_t dim) {
    std::vector<T> norms(n);
    for (size_t i = 0; i < n; ++i) {
      const T* row = data + i * dim;
      norms[i] = std::inner_product(row, row + dim, row, T{0});
    }
    return norms;
  }

}  // namespace

// =============================================================================
// Float Specialization
// =============================================================================

template <> void CudaClusterBackend<float>::free_memory() {
  if (graph_exec_) {
    cudaGraphExecDestroy(graph_exec_);
    graph_exec_ = nullptr;
  }
  if (graph_) {
    cudaGraphDestroy(graph_);
    graph_ = nullptr;
  }
  for (int i = 0; i < kNumPipelineStages; ++i) {
    if (pipeline_cublas_[i]) {
      cublasDestroy(pipeline_cublas_[i]);
      pipeline_cublas_[i] = nullptr;
    }
    if (stages_[i].event) {
      cudaEventDestroy(stages_[i].event);
      stages_[i].event = nullptr;
    }
    if (stages_[i].stream) {
      cudaStreamDestroy(stages_[i].stream);
      stages_[i].stream = nullptr;
    }
    stages_[i].d_queries.reset();
    stages_[i].d_norms.reset();
    stages_[i].d_dots.reset();
    stages_[i].d_idx.reset();
    stages_[i].d_dist.reset();
    stages_[i].h_queries.reset();
    stages_[i].h_idx.reset();
    stages_[i].h_dist.reset();
    stages_[i].capacity = 0;
  }
  pipeline_initialized_ = false;
}

template <> CudaClusterBackend<float>::CudaClusterBackend() {
  try {
    NORDLYS_CUDA_CHECK(cudaStreamCreate(&stream_));
    NORDLYS_CUBLAS_CHECK(cublasCreate(&cublas_));
    NORDLYS_CUBLAS_CHECK(cublasSetStream(cublas_, stream_));
  } catch (...) {
    if (cublas_) {
      cublasDestroy(cublas_);
      cublas_ = nullptr;
    }
    if (stream_) {
      cudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
    throw;
  }
}

template <> CudaClusterBackend<float>::~CudaClusterBackend() {
  free_memory();
  if (cublas_) cublasDestroy(cublas_);
  if (stream_) cudaStreamDestroy(stream_);
}

template <> void CudaClusterBackend<float>::capture_graph() {
  if (graph_exec_) {
    cudaGraphExecDestroy(graph_exec_);
    graph_exec_ = nullptr;
  }
  if (graph_) {
    cudaGraphDestroy(graph_);
    graph_ = nullptr;
  }

  NORDLYS_CUDA_CHECK(cudaStreamSynchronize(stream_));
  NORDLYS_CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));

  NORDLYS_CUDA_CHECK(cudaMemcpyAsync(d_embedding_.get(), h_embedding_.get(), dim_ * sizeof(float),
                                     cudaMemcpyHostToDevice, stream_));

  float alpha = 1.0f, beta = 0.0f;
  NORDLYS_CUBLAS_CHECK(cublasSgemv(cublas_, CUBLAS_OP_N, n_clusters_, dim_, &alpha,
                                   d_centroids_.get(), n_clusters_, d_embedding_.get(), 1, &beta,
                                   d_dots_.get(), 1));

  int fused_block = std::max((dim_ <= 512) ? 128 : 256, (n_clusters_ <= 128) ? 64 : 128);
  size_t shared_mem = fused_l2_argmin_shared_mem_size<float>();
  fused_l2_argmin_with_norm<float><<<1, fused_block, shared_mem, stream_>>>(
      d_embedding_.get(), d_centroid_norms_.get(), d_dots_.get(), n_clusters_, dim_,
      d_best_idx_.get(), d_best_dist_.get());
  NORDLYS_CUDA_CHECK(cudaGetLastError());

  NORDLYS_CUDA_CHECK(cudaMemcpyAsync(h_best_idx_.get(), d_best_idx_.get(), sizeof(int),
                                     cudaMemcpyDeviceToHost, stream_));
  NORDLYS_CUDA_CHECK(cudaMemcpyAsync(h_best_dist_.get(), d_best_dist_.get(), sizeof(float),
                                     cudaMemcpyDeviceToHost, stream_));

  NORDLYS_CUDA_CHECK(cudaStreamEndCapture(stream_, &graph_));
  NORDLYS_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0));

  graph_valid_ = true;
}

template <>
void CudaClusterBackend<float>::load_centroids(const float* data, size_t n_clusters, size_t dim) {
  if (n_clusters <= 0 || dim <= 0) {
    throw std::invalid_argument("n_clusters and dim must be positive");
  }

  auto nc = static_cast<size_t>(n_clusters);
  auto d = static_cast<size_t>(dim);

  if (nc > SIZE_MAX / d) {
    throw std::invalid_argument("n_clusters * dim would overflow");
  }

  free_memory();

  n_clusters_ = n_clusters;
  dim_ = dim;

  d_centroids_.reset(nc * d);
  d_centroid_norms_.reset(nc);
  d_embedding_.reset(d);
  d_embed_norm_.reset(1);
  d_dots_.reset(nc);
  d_best_idx_.reset(1);
  d_best_dist_.reset(1);

  h_embedding_.reset(d);
  h_best_idx_.reset(1);
  h_best_dist_.reset(1);

  auto centroids_col = to_col_major(data, n_clusters, dim);
  auto norms = compute_squared_norms_cpu(data, n_clusters, dim);

  NORDLYS_CUDA_CHECK(cudaMemcpy(d_centroids_.get(), centroids_col.data(),
                                n_clusters * dim * sizeof(float), cudaMemcpyHostToDevice));
  NORDLYS_CUDA_CHECK(cudaMemcpy(d_centroid_norms_.get(), norms.data(), n_clusters * sizeof(float),
                                cudaMemcpyHostToDevice));

  std::fill_n(h_embedding_.get(), dim, 1.0f);
  capture_graph();
}

template <> std::pair<int, float> CudaClusterBackend<float>::assign(EmbeddingView<float> view) {
  if (n_clusters_ == 0) {
    return {-1, 0.0f};
  }

  if (view.dim != static_cast<size_t>(dim_)) {
    throw std::invalid_argument("dimension mismatch in assign");
  }

  if (!graph_valid_ || !graph_exec_) {
    throw std::runtime_error("CUDA graph not initialized - call load_centroids first");
  }

  return std::visit(
      overloaded{[&](CpuDevice) -> std::pair<int, float> {
                   // CPU input: copy to GPU (existing path)
                   std::memcpy(h_embedding_.get(), view.data, view.dim * sizeof(float));
                   NORDLYS_CUDA_CHECK(cudaMemcpyAsync(d_embedding_.get(), h_embedding_.get(),
                                                      view.dim * sizeof(float),
                                                      cudaMemcpyHostToDevice, stream_));

                   // Use existing graph (expects data in d_embedding_)
                   NORDLYS_CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream_));
                   NORDLYS_CUDA_CHECK(cudaStreamSynchronize(stream_));

                   return {*h_best_idx_.get(), *h_best_dist_.get()};
                 },
                 [&](CudaDevice d) -> std::pair<int, float> {
                   // GPU input: use pointer directly
                   // Ensure we're on the correct device
                   int current_device;
                   cudaGetDevice(&current_device);
                   if (current_device != d.id) {
                     NORDLYS_CUDA_CHECK(cudaSetDevice(d.id));
                   }

                   // Copy directly from input GPU pointer to our GPU buffer
                   NORDLYS_CUDA_CHECK(cudaMemcpyAsync(d_embedding_.get(), view.data,
                                                      view.dim * sizeof(float),
                                                      cudaMemcpyDeviceToDevice, stream_));

                   // Use existing graph
                   NORDLYS_CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream_));
                   NORDLYS_CUDA_CHECK(cudaStreamSynchronize(stream_));

                   return {*h_best_idx_.get(), *h_best_dist_.get()};
                 }},
      view.device);
}

template <> void CudaClusterBackend<float>::init_pipeline() {
  if (pipeline_initialized_) return;
  for (int i = 0; i < kNumPipelineStages; ++i) {
    NORDLYS_CUDA_CHECK(cudaStreamCreate(&stages_[i].stream));
    NORDLYS_CUDA_CHECK(cudaEventCreate(&stages_[i].event));
    NORDLYS_CUBLAS_CHECK(cublasCreate(&pipeline_cublas_[i]));
    NORDLYS_CUBLAS_CHECK(cublasSetStream(pipeline_cublas_[i], stages_[i].stream));
  }
  pipeline_initialized_ = true;
}

template <> void CudaClusterBackend<float>::ensure_stage_capacity(int stage_idx, int count) {
  auto& stage = stages_[stage_idx];
  if (count <= stage.capacity) return;

  int new_cap = count + count / 2;

  stage.d_queries.reset(new_cap * dim_);
  stage.d_norms.reset(new_cap);
  stage.d_dots.reset(new_cap * n_clusters_);
  stage.d_idx.reset(new_cap);
  stage.d_dist.reset(new_cap);

  stage.h_queries.reset(new_cap * dim_);
  stage.h_idx.reset(new_cap);
  stage.h_dist.reset(new_cap);

  stage.capacity = new_cap;
}

// Forward declarations for batch processing helpers
template <> std::vector<std::pair<int, float>> CudaClusterBackend<float>::assign_batch_from_host(
    EmbeddingBatchView<float> view);

template <> std::vector<std::pair<int, float>> CudaClusterBackend<float>::assign_batch_from_device(
    EmbeddingBatchView<float> view);

template <> std::vector<std::pair<int, float>> CudaClusterBackend<float>::assign_batch(
    EmbeddingBatchView<float> view) {
  if (view.count == 0) return {};
  if (view.count == 1) {
    EmbeddingView<float> single_view{view.data, view.dim, view.device};
    return {assign(single_view)};
  }
  if (view.dim != static_cast<size_t>(dim_)) {
    throw std::invalid_argument("dimension mismatch");
  }

  return std::visit(overloaded{[&](CpuDevice) -> std::vector<std::pair<int, float>> {
                                 return assign_batch_from_host(view);
                               },
                               [&](CudaDevice d) -> std::vector<std::pair<int, float>> {
                                 // Ensure we're on the correct device
                                 int current_device;
                                 cudaGetDevice(&current_device);
                                 if (current_device != d.id) {
                                   NORDLYS_CUDA_CHECK(cudaSetDevice(d.id));
                                 }
                                 return assign_batch_from_device(view);
                               }},
                    view.device);
}

template <> std::vector<std::pair<int, float>> CudaClusterBackend<float>::assign_batch_from_device(
    EmbeddingBatchView<float> view) {
  init_pipeline();

  constexpr int kMinChunkSize = 64;
  int chunk_size = std::max(
      kMinChunkSize, (static_cast<int>(view.count) + kNumPipelineStages - 1) / kNumPipelineStages);
  int num_chunks = (static_cast<int>(view.count) + chunk_size - 1) / chunk_size;

  for (int i = 0; i < kNumPipelineStages; ++i) {
    ensure_stage_capacity(i, chunk_size);
  }

  std::vector<std::pair<int, float>> results(view.count);
  int fused_threads = std::max((dim_ <= 512) ? 128 : 256, (n_clusters_ <= 128) ? 64 : 128);
  size_t shared_mem = fused_l2_argmin_shared_mem_size<float>();
  float alpha = 1.0f, beta = 0.0f;

  for (int chunk = 0; chunk < num_chunks; ++chunk) {
    int stage_idx = chunk % kNumPipelineStages;
    auto& stage = stages_[stage_idx];
    auto& cublas = pipeline_cublas_[stage_idx];

    int offset = chunk * chunk_size;
    int this_count = std::min(chunk_size, static_cast<int>(view.count) - offset);
    const float* src = view.data + offset * static_cast<int>(view.dim);

    if (chunk >= kNumPipelineStages) {
      int prev_chunk = chunk - kNumPipelineStages;
      int prev_stage = prev_chunk % kNumPipelineStages;
      auto& prev = stages_[prev_stage];

      NORDLYS_CUDA_CHECK(cudaEventSynchronize(prev.event));

      int prev_offset = prev_chunk * chunk_size;
      int prev_count = std::min(chunk_size, static_cast<int>(view.count) - prev_offset);
      for (int i = 0; i < prev_count; ++i) {
        results[prev_offset + i] = {prev.h_idx.get()[i], prev.h_dist.get()[i]};
      }
    }

    // Device-to-device copy
    NORDLYS_CUDA_CHECK(cudaMemcpyAsync(stage.d_queries.get(), src,
                                       this_count * dim_ * sizeof(float), cudaMemcpyDeviceToDevice,
                                       stage.stream));

    NORDLYS_CUBLAS_CHECK(cublasSgemm(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, n_clusters_, this_count, dim_, &alpha, d_centroids_.get(),
        n_clusters_, stage.d_queries.get(), dim_, &beta, stage.d_dots.get(), n_clusters_));

    batch_l2_argmin_fused<<<this_count, fused_threads, shared_mem, stage.stream>>>(
        stage.d_queries.get(), d_centroid_norms_.get(), stage.d_dots.get(), this_count, n_clusters_,
        dim_, stage.d_idx.get(), stage.d_dist.get());

    NORDLYS_CUDA_CHECK(cudaMemcpyAsync(stage.h_idx.get(), stage.d_idx.get(),
                                       this_count * sizeof(int), cudaMemcpyDeviceToHost,
                                       stage.stream));
    NORDLYS_CUDA_CHECK(cudaMemcpyAsync(stage.h_dist.get(), stage.d_dist.get(),
                                       this_count * sizeof(float), cudaMemcpyDeviceToHost,
                                       stage.stream));

    NORDLYS_CUDA_CHECK(cudaEventRecord(stage.event, stage.stream));
  }

  for (int tail = std::max(0, num_chunks - kNumPipelineStages); tail < num_chunks; ++tail) {
    int stage_idx = tail % kNumPipelineStages;
    auto& stage = stages_[stage_idx];

    NORDLYS_CUDA_CHECK(cudaEventSynchronize(stage.event));

    int offset = tail * chunk_size;
    int this_count = std::min(chunk_size, static_cast<int>(view.count) - offset);
    for (int i = 0; i < this_count; ++i) {
      results[offset + i] = {stage.h_idx.get()[i], stage.h_dist.get()[i]};
    }
  }

  return results;
}

template <> std::vector<std::pair<int, float>> CudaClusterBackend<float>::assign_batch_from_host(
    EmbeddingBatchView<float> view) {
  init_pipeline();

  constexpr int kMinChunkSize = 64;
  int chunk_size = std::max(
      kMinChunkSize, (static_cast<int>(view.count) + kNumPipelineStages - 1) / kNumPipelineStages);
  int num_chunks = (static_cast<int>(view.count) + chunk_size - 1) / chunk_size;

  for (int i = 0; i < kNumPipelineStages; ++i) {
    ensure_stage_capacity(i, chunk_size);
  }

  std::vector<std::pair<int, float>> results(view.count);
  int fused_threads = std::max((dim_ <= 512) ? 128 : 256, (n_clusters_ <= 128) ? 64 : 128);
  size_t shared_mem = fused_l2_argmin_shared_mem_size<float>();
  float alpha = 1.0f, beta = 0.0f;

  for (int chunk = 0; chunk < num_chunks; ++chunk) {
    int stage_idx = chunk % kNumPipelineStages;
    auto& stage = stages_[stage_idx];
    auto& cublas = pipeline_cublas_[stage_idx];

    int offset = chunk * chunk_size;
    int this_count = std::min(chunk_size, static_cast<int>(view.count) - offset);
    const float* src = view.data + offset * static_cast<int>(view.dim);

    if (chunk >= kNumPipelineStages) {
      int prev_chunk = chunk - kNumPipelineStages;
      int prev_stage = prev_chunk % kNumPipelineStages;
      auto& prev = stages_[prev_stage];

      NORDLYS_CUDA_CHECK(cudaEventSynchronize(prev.event));

      int prev_offset = prev_chunk * chunk_size;
      int prev_count = std::min(chunk_size, static_cast<int>(view.count) - prev_offset);
      for (int i = 0; i < prev_count; ++i) {
        results[prev_offset + i] = {prev.h_idx.get()[i], prev.h_dist.get()[i]};
      }
    }

    NORDLYS_CUDA_CHECK(cudaMemcpyAsync(stage.d_queries.get(), src,
                                       this_count * dim_ * sizeof(float), cudaMemcpyHostToDevice,
                                       stage.stream));

    NORDLYS_CUBLAS_CHECK(cublasSgemm(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, n_clusters_, this_count, dim_, &alpha, d_centroids_.get(),
        n_clusters_, stage.d_queries.get(), dim_, &beta, stage.d_dots.get(), n_clusters_));

    batch_l2_argmin_fused<<<this_count, fused_threads, shared_mem, stage.stream>>>(
        stage.d_queries.get(), d_centroid_norms_.get(), stage.d_dots.get(), this_count, n_clusters_,
        dim_, stage.d_idx.get(), stage.d_dist.get());

    NORDLYS_CUDA_CHECK(cudaMemcpyAsync(stage.h_idx.get(), stage.d_idx.get(),
                                       this_count * sizeof(int), cudaMemcpyDeviceToHost,
                                       stage.stream));
    NORDLYS_CUDA_CHECK(cudaMemcpyAsync(stage.h_dist.get(), stage.d_dist.get(),
                                       this_count * sizeof(float), cudaMemcpyDeviceToHost,
                                       stage.stream));

    NORDLYS_CUDA_CHECK(cudaEventRecord(stage.event, stage.stream));
  }

  for (int tail = std::max(0, num_chunks - kNumPipelineStages); tail < num_chunks; ++tail) {
    int stage_idx = tail % kNumPipelineStages;
    auto& stage = stages_[stage_idx];

    NORDLYS_CUDA_CHECK(cudaEventSynchronize(stage.event));

    int offset = tail * chunk_size;
    int this_count = std::min(chunk_size, static_cast<int>(view.count) - offset);
    for (int i = 0; i < this_count; ++i) {
      results[offset + i] = {stage.h_idx.get()[i], stage.h_dist.get()[i]};
    }
  }

  return results;
}

// =============================================================================
// Double Specialization
// =============================================================================

template <> void CudaClusterBackend<double>::free_memory() {
  if (graph_exec_) {
    cudaGraphExecDestroy(graph_exec_);
    graph_exec_ = nullptr;
  }
  if (graph_) {
    cudaGraphDestroy(graph_);
    graph_ = nullptr;
  }
  for (int i = 0; i < kNumPipelineStages; ++i) {
    if (pipeline_cublas_[i]) {
      cublasDestroy(pipeline_cublas_[i]);
      pipeline_cublas_[i] = nullptr;
    }
    if (stages_[i].event) {
      cudaEventDestroy(stages_[i].event);
      stages_[i].event = nullptr;
    }
    if (stages_[i].stream) {
      cudaStreamDestroy(stages_[i].stream);
      stages_[i].stream = nullptr;
    }
    stages_[i].d_queries.reset();
    stages_[i].d_norms.reset();
    stages_[i].d_dots.reset();
    stages_[i].d_idx.reset();
    stages_[i].d_dist.reset();
    stages_[i].h_queries.reset();
    stages_[i].h_idx.reset();
    stages_[i].h_dist.reset();
    stages_[i].capacity = 0;
  }
  pipeline_initialized_ = false;
}

template <> CudaClusterBackend<double>::CudaClusterBackend() {
  try {
    NORDLYS_CUDA_CHECK(cudaStreamCreate(&stream_));
    NORDLYS_CUBLAS_CHECK(cublasCreate(&cublas_));
    NORDLYS_CUBLAS_CHECK(cublasSetStream(cublas_, stream_));
  } catch (...) {
    if (cublas_) {
      cublasDestroy(cublas_);
      cublas_ = nullptr;
    }
    if (stream_) {
      cudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
    throw;
  }
}

template <> CudaClusterBackend<double>::~CudaClusterBackend() {
  free_memory();
  if (cublas_) cublasDestroy(cublas_);
  if (stream_) cudaStreamDestroy(stream_);
}

template <> void CudaClusterBackend<double>::capture_graph() {
  if (graph_exec_) {
    cudaGraphExecDestroy(graph_exec_);
    graph_exec_ = nullptr;
  }
  if (graph_) {
    cudaGraphDestroy(graph_);
    graph_ = nullptr;
  }

  NORDLYS_CUDA_CHECK(cudaStreamSynchronize(stream_));
  NORDLYS_CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));

  NORDLYS_CUDA_CHECK(cudaMemcpyAsync(d_embedding_.get(), h_embedding_.get(), dim_ * sizeof(double),
                                     cudaMemcpyHostToDevice, stream_));

  double alpha = 1.0, beta = 0.0;
  NORDLYS_CUBLAS_CHECK(cublasDgemv(cublas_, CUBLAS_OP_N, n_clusters_, dim_, &alpha,
                                   d_centroids_.get(), n_clusters_, d_embedding_.get(), 1, &beta,
                                   d_dots_.get(), 1));

  int fused_block = std::max((dim_ <= 512) ? 128 : 256, (n_clusters_ <= 128) ? 64 : 128);
  size_t shared_mem = fused_l2_argmin_shared_mem_size<double>();
  fused_l2_argmin_with_norm<double><<<1, fused_block, shared_mem, stream_>>>(
      d_embedding_.get(), d_centroid_norms_.get(), d_dots_.get(), n_clusters_, dim_,
      d_best_idx_.get(), d_best_dist_.get());
  NORDLYS_CUDA_CHECK(cudaGetLastError());

  NORDLYS_CUDA_CHECK(cudaMemcpyAsync(h_best_idx_.get(), d_best_idx_.get(), sizeof(int),
                                     cudaMemcpyDeviceToHost, stream_));
  NORDLYS_CUDA_CHECK(cudaMemcpyAsync(h_best_dist_.get(), d_best_dist_.get(), sizeof(double),
                                     cudaMemcpyDeviceToHost, stream_));

  NORDLYS_CUDA_CHECK(cudaStreamEndCapture(stream_, &graph_));
  NORDLYS_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0));

  graph_valid_ = true;
}

template <>
void CudaClusterBackend<double>::load_centroids(const double* data, size_t n_clusters, size_t dim) {
  if (n_clusters == 0 || dim == 0) {
    throw std::invalid_argument("n_clusters and dim must be positive");
  }

  if (n_clusters > SIZE_MAX / dim) {
    throw std::invalid_argument("n_clusters * dim would overflow");
  }

  free_memory();

  n_clusters_ = static_cast<int>(n_clusters);
  dim_ = static_cast<int>(dim);

  auto nc = static_cast<size_t>(n_clusters);
  auto d = static_cast<size_t>(dim);

  d_centroids_.reset(nc * d);
  d_centroid_norms_.reset(nc);
  d_embedding_.reset(d);
  d_embed_norm_.reset(1);
  d_dots_.reset(nc);
  d_best_idx_.reset(1);
  d_best_dist_.reset(1);

  h_embedding_.reset(d);
  h_best_idx_.reset(1);
  h_best_dist_.reset(1);

  auto centroids_col = to_col_major(data, n_clusters, dim);
  auto norms = compute_squared_norms_cpu(data, n_clusters, dim);

  NORDLYS_CUDA_CHECK(cudaMemcpy(d_centroids_.get(), centroids_col.data(),
                                n_clusters * dim * sizeof(double), cudaMemcpyHostToDevice));
  NORDLYS_CUDA_CHECK(cudaMemcpy(d_centroid_norms_.get(), norms.data(), n_clusters * sizeof(double),
                                cudaMemcpyHostToDevice));

  std::fill_n(h_embedding_.get(), dim, 1.0);
  capture_graph();
}

template <> std::pair<int, double> CudaClusterBackend<double>::assign(EmbeddingView<double> view) {
  if (n_clusters_ == 0) {
    return {-1, 0.0};
  }

  if (view.dim != static_cast<size_t>(dim_)) {
    throw std::invalid_argument("dimension mismatch in assign");
  }

  if (!graph_valid_ || !graph_exec_) {
    throw std::runtime_error("CUDA graph not initialized - call load_centroids first");
  }

  return std::visit(
      overloaded{[&](CpuDevice) -> std::pair<int, double> {
                   // CPU input: copy to GPU (existing path)
                   std::memcpy(h_embedding_.get(), view.data, view.dim * sizeof(double));
                   NORDLYS_CUDA_CHECK(cudaMemcpyAsync(d_embedding_.get(), h_embedding_.get(),
                                                      view.dim * sizeof(double),
                                                      cudaMemcpyHostToDevice, stream_));

                   // Use existing graph
                   NORDLYS_CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream_));
                   NORDLYS_CUDA_CHECK(cudaStreamSynchronize(stream_));

                   return {*h_best_idx_.get(), *h_best_dist_.get()};
                 },
                 [&](CudaDevice d) -> std::pair<int, double> {
                   // GPU input: use pointer directly
                   int current_device;
                   cudaGetDevice(&current_device);
                   if (current_device != d.id) {
                     NORDLYS_CUDA_CHECK(cudaSetDevice(d.id));
                   }

                   // Copy directly from input GPU pointer to our GPU buffer
                   NORDLYS_CUDA_CHECK(cudaMemcpyAsync(d_embedding_.get(), view.data,
                                                      view.dim * sizeof(double),
                                                      cudaMemcpyDeviceToDevice, stream_));

                   // Use existing graph
                   NORDLYS_CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream_));
                   NORDLYS_CUDA_CHECK(cudaStreamSynchronize(stream_));

                   return {*h_best_idx_.get(), *h_best_dist_.get()};
                 }},
      view.device);
}

template <> void CudaClusterBackend<double>::init_pipeline() {
  if (pipeline_initialized_) return;
  for (int i = 0; i < kNumPipelineStages; ++i) {
    NORDLYS_CUDA_CHECK(cudaStreamCreate(&stages_[i].stream));
    NORDLYS_CUDA_CHECK(cudaEventCreate(&stages_[i].event));
    NORDLYS_CUBLAS_CHECK(cublasCreate(&pipeline_cublas_[i]));
    NORDLYS_CUBLAS_CHECK(cublasSetStream(pipeline_cublas_[i], stages_[i].stream));
  }
  pipeline_initialized_ = true;
}

template <> void CudaClusterBackend<double>::ensure_stage_capacity(int stage_idx, int count) {
  auto& stage = stages_[stage_idx];
  if (count <= stage.capacity) return;

  int new_cap = count + count / 2;

  stage.d_queries.reset(new_cap * dim_);
  stage.d_norms.reset(new_cap);
  stage.d_dots.reset(new_cap * n_clusters_);
  stage.d_idx.reset(new_cap);
  stage.d_dist.reset(new_cap);

  stage.h_queries.reset(new_cap * dim_);
  stage.h_idx.reset(new_cap);
  stage.h_dist.reset(new_cap);

  stage.capacity = new_cap;
}

// Forward declarations for batch processing helpers
template <> std::vector<std::pair<int, double>> CudaClusterBackend<double>::assign_batch_from_host(
    EmbeddingBatchView<double> view);

template <>
std::vector<std::pair<int, double>> CudaClusterBackend<double>::assign_batch_from_device(
    EmbeddingBatchView<double> view);

template <> std::vector<std::pair<int, double>> CudaClusterBackend<double>::assign_batch(
    EmbeddingBatchView<double> view) {
  if (view.count == 0) return {};
  if (view.count == 1) {
    EmbeddingView<double> single_view{view.data, view.dim, view.device};
    return {assign(single_view)};
  }
  if (view.dim != static_cast<size_t>(dim_)) {
    throw std::invalid_argument("dimension mismatch");
  }

  return std::visit(overloaded{[&](CpuDevice) -> std::vector<std::pair<int, double>> {
                                 return assign_batch_from_host(view);
                               },
                               [&](CudaDevice d) -> std::vector<std::pair<int, double>> {
                                 int current_device;
                                 cudaGetDevice(&current_device);
                                 if (current_device != d.id) {
                                   NORDLYS_CUDA_CHECK(cudaSetDevice(d.id));
                                 }
                                 return assign_batch_from_device(view);
                               }},
                    view.device);
}

template <> std::vector<std::pair<int, double>> CudaClusterBackend<double>::assign_batch_from_host(
    EmbeddingBatchView<double> view) {
  init_pipeline();

  constexpr int kMinChunkSize = 64;
  int chunk_size = std::max(
      kMinChunkSize, (static_cast<int>(view.count) + kNumPipelineStages - 1) / kNumPipelineStages);
  int num_chunks = (static_cast<int>(view.count) + chunk_size - 1) / chunk_size;

  for (int i = 0; i < kNumPipelineStages; ++i) {
    ensure_stage_capacity(i, chunk_size);
  }

  std::vector<std::pair<int, double>> results(view.count);
  int fused_threads = std::max((dim_ <= 512) ? 128 : 256, (n_clusters_ <= 128) ? 64 : 128);
  size_t shared_mem = fused_l2_argmin_shared_mem_size<double>();
  double alpha = 1.0, beta = 0.0;

  for (int chunk = 0; chunk < num_chunks; ++chunk) {
    int stage_idx = chunk % kNumPipelineStages;
    auto& stage = stages_[stage_idx];
    auto& cublas = pipeline_cublas_[stage_idx];

    int offset = chunk * chunk_size;
    int this_count = std::min(chunk_size, static_cast<int>(view.count) - offset);
    const double* src = view.data + offset * static_cast<int>(view.dim);

    if (chunk >= kNumPipelineStages) {
      int prev_chunk = chunk - kNumPipelineStages;
      int prev_stage = prev_chunk % kNumPipelineStages;
      auto& prev = stages_[prev_stage];

      NORDLYS_CUDA_CHECK(cudaEventSynchronize(prev.event));

      int prev_offset = prev_chunk * chunk_size;
      int prev_count = std::min(chunk_size, static_cast<int>(view.count) - prev_offset);
      for (int i = 0; i < prev_count; ++i) {
        results[prev_offset + i] = {prev.h_idx.get()[i], prev.h_dist.get()[i]};
      }
    }

    NORDLYS_CUDA_CHECK(cudaMemcpyAsync(stage.d_queries.get(), src,
                                       this_count * dim_ * sizeof(double), cudaMemcpyHostToDevice,
                                       stage.stream));

    NORDLYS_CUBLAS_CHECK(cublasDgemm(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, n_clusters_, this_count, dim_, &alpha, d_centroids_.get(),
        n_clusters_, stage.d_queries.get(), dim_, &beta, stage.d_dots.get(), n_clusters_));

    batch_l2_argmin_fused<<<this_count, fused_threads, shared_mem, stage.stream>>>(
        stage.d_queries.get(), d_centroid_norms_.get(), stage.d_dots.get(), this_count, n_clusters_,
        dim_, stage.d_idx.get(), stage.d_dist.get());

    NORDLYS_CUDA_CHECK(cudaMemcpyAsync(stage.h_idx.get(), stage.d_idx.get(),
                                       this_count * sizeof(int), cudaMemcpyDeviceToHost,
                                       stage.stream));
    NORDLYS_CUDA_CHECK(cudaMemcpyAsync(stage.h_dist.get(), stage.d_dist.get(),
                                       this_count * sizeof(double), cudaMemcpyDeviceToHost,
                                       stage.stream));

    NORDLYS_CUDA_CHECK(cudaEventRecord(stage.event, stage.stream));
  }

  for (int tail = std::max(0, num_chunks - kNumPipelineStages); tail < num_chunks; ++tail) {
    int stage_idx = tail % kNumPipelineStages;
    auto& stage = stages_[stage_idx];

    NORDLYS_CUDA_CHECK(cudaEventSynchronize(stage.event));

    int offset = tail * chunk_size;
    int this_count = std::min(chunk_size, static_cast<int>(view.count) - offset);
    for (int i = 0; i < this_count; ++i) {
      results[offset + i] = {stage.h_idx.get()[i], stage.h_dist.get()[i]};
    }
  }

  return results;
}

template <>
std::vector<std::pair<int, double>> CudaClusterBackend<double>::assign_batch_from_device(
    EmbeddingBatchView<double> view) {
  init_pipeline();

  constexpr int kMinChunkSize = 64;
  int chunk_size = std::max(
      kMinChunkSize, (static_cast<int>(view.count) + kNumPipelineStages - 1) / kNumPipelineStages);
  int num_chunks = (static_cast<int>(view.count) + chunk_size - 1) / chunk_size;

  for (int i = 0; i < kNumPipelineStages; ++i) {
    ensure_stage_capacity(i, chunk_size);
  }

  std::vector<std::pair<int, double>> results(view.count);
  int fused_threads = std::max((dim_ <= 512) ? 128 : 256, (n_clusters_ <= 128) ? 64 : 128);
  size_t shared_mem = fused_l2_argmin_shared_mem_size<double>();
  double alpha = 1.0, beta = 0.0;

  for (int chunk = 0; chunk < num_chunks; ++chunk) {
    int stage_idx = chunk % kNumPipelineStages;
    auto& stage = stages_[stage_idx];
    auto& cublas = pipeline_cublas_[stage_idx];

    int offset = chunk * chunk_size;
    int this_count = std::min(chunk_size, static_cast<int>(view.count) - offset);
    const double* src = view.data + offset * static_cast<int>(view.dim);

    if (chunk >= kNumPipelineStages) {
      int prev_chunk = chunk - kNumPipelineStages;
      int prev_stage = prev_chunk % kNumPipelineStages;
      auto& prev = stages_[prev_stage];

      NORDLYS_CUDA_CHECK(cudaEventSynchronize(prev.event));

      int prev_offset = prev_chunk * chunk_size;
      int prev_count = std::min(chunk_size, static_cast<int>(view.count) - prev_offset);
      for (int i = 0; i < prev_count; ++i) {
        results[prev_offset + i] = {prev.h_idx.get()[i], prev.h_dist.get()[i]};
      }
    }

    // Device-to-device copy
    NORDLYS_CUDA_CHECK(cudaMemcpyAsync(stage.d_queries.get(), src,
                                       this_count * dim_ * sizeof(double), cudaMemcpyDeviceToDevice,
                                       stage.stream));

    NORDLYS_CUBLAS_CHECK(cublasDgemm(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, n_clusters_, this_count, dim_, &alpha, d_centroids_.get(),
        n_clusters_, stage.d_queries.get(), dim_, &beta, stage.d_dots.get(), n_clusters_));

    batch_l2_argmin_fused<<<this_count, fused_threads, shared_mem, stage.stream>>>(
        stage.d_queries.get(), d_centroid_norms_.get(), stage.d_dots.get(), this_count, n_clusters_,
        dim_, stage.d_idx.get(), stage.d_dist.get());

    NORDLYS_CUDA_CHECK(cudaMemcpyAsync(stage.h_idx.get(), stage.d_idx.get(),
                                       this_count * sizeof(int), cudaMemcpyDeviceToHost,
                                       stage.stream));
    NORDLYS_CUDA_CHECK(cudaMemcpyAsync(stage.h_dist.get(), stage.d_dist.get(),
                                       this_count * sizeof(double), cudaMemcpyDeviceToHost,
                                       stage.stream));

    NORDLYS_CUDA_CHECK(cudaEventRecord(stage.event, stage.stream));
  }

  for (int tail = std::max(0, num_chunks - kNumPipelineStages); tail < num_chunks; ++tail) {
    int stage_idx = tail % kNumPipelineStages;
    auto& stage = stages_[stage_idx];

    NORDLYS_CUDA_CHECK(cudaEventSynchronize(stage.event));

    int offset = tail * chunk_size;
    int this_count = std::min(chunk_size, static_cast<int>(view.count) - offset);
    for (int i = 0; i < this_count; ++i) {
      results[offset + i] = {stage.h_idx.get()[i], stage.h_dist.get()[i]};
    }
  }

  return results;
}

#endif  // NORDLYS_HAS_CUDA
