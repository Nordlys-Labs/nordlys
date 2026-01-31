#ifdef NORDLYS_HAS_CUDA

#  include <climits>
#  include <cmath>
#  include <cstring>
#  include <nordlys/clustering/cluster_cuda.hpp>
#  include <nordlys/clustering/cuda/common.cuh>
#  include <nordlys/clustering/cuda/distance.cuh>
#  include <nordlys/clustering/cuda/utils.cuh>
#  include <numeric>
#  include <span>
#  include <stdexcept>
#  include <string>
#  include <vector>

#ifdef USE_NVTX
#  include <nvtx3/nvToolsExt.h>
#  define NVTX_RANGE_PUSH(name) nvtxRangePushA(name)
#  define NVTX_RANGE_POP() nvtxRangePop()
#else
#  define NVTX_RANGE_PUSH(name) ((void)0)
#  define NVTX_RANGE_POP() ((void)0)
#endif

namespace nordlys::clustering {

// =============================================================================
// CudaClusterBackend Implementation
// =============================================================================

void CudaClusterBackend::free_memory() {
  if (graph_exec_) {
    cudaGraphExecDestroy(graph_exec_);
    graph_exec_ = nullptr;
  }
  if (graph_) {
    cudaGraphDestroy(graph_);
    graph_ = nullptr;
  }
  
  // Free main stream allocations
  if (stream_) {
    d_centroids_.free_async(stream_);
    d_centroids_row_.free_async(stream_);
    d_centroid_norms_.free_async(stream_);
    d_embedding_.free_async(stream_);
    d_embed_norm_.free_async(stream_);
    d_dots_.free_async(stream_);
    d_best_idx_.free_async(stream_);
    d_best_dist_.free_async(stream_);
  }
  for (int i = 0; i < kNumPipelineStages; ++i) {
    if (pipeline_cublas_[i]) {
      cublasDestroy(pipeline_cublas_[i]);
      pipeline_cublas_[i] = nullptr;
    }
    // Free device memory before destroying streams
    if (stages_[i].stream) {
      stages_[i].d_queries.free_async(stages_[i].stream);
      stages_[i].d_norms.free_async(stages_[i].stream);
      stages_[i].d_dots.free_async(stages_[i].stream);
      stages_[i].d_results.free_async(stages_[i].stream);
    }
    stages_[i].h_queries.reset();
    stages_[i].h_results.reset();
    if (stages_[i].event) {
      cudaEventDestroy(stages_[i].event);
      stages_[i].event = nullptr;
    }
    if (stages_[i].stream) {
      cudaStreamDestroy(stages_[i].stream);
      stages_[i].stream = nullptr;
    }
    stages_[i].capacity = 0;
  }
  pipeline_initialized_ = false;
}

CudaClusterBackend::CudaClusterBackend() {
  try {
    NORDLYS_CUDA_CHECK(cudaStreamCreate(&stream_));
    NORDLYS_CUBLAS_CHECK(cublasCreate(&cublas_));
    NORDLYS_CUBLAS_CHECK(cublasSetStream(cublas_, stream_));
    memory_pool_ = std::make_unique<cuda::MemoryPool>();
  } catch (...) {
    memory_pool_.reset();
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

CudaClusterBackend::~CudaClusterBackend() {
  free_memory();
  memory_pool_.reset();
  if (cublas_) cublasDestroy(cublas_);
  if (stream_) cudaStreamDestroy(stream_);
}

void CudaClusterBackend::capture_graph() {
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

  int block_size = 256;
  size_t shared_mem = cuda::reduction_shared_mem_size<float>();
  cuda::find_nearest_centroid_with_dots<float><<<1, block_size, shared_mem, stream_>>>(
      d_embedding_.get(), d_centroid_norms_.get(), d_dots_.get(), 
      n_clusters_, dim_, d_best_idx_.get(), d_best_dist_.get());
  NORDLYS_CUDA_CHECK(cudaGetLastError());

  NORDLYS_CUDA_CHECK(cudaMemcpyAsync(h_best_idx_.get(), d_best_idx_.get(), sizeof(int),
                                     cudaMemcpyDeviceToHost, stream_));
  NORDLYS_CUDA_CHECK(cudaMemcpyAsync(h_best_dist_.get(), d_best_dist_.get(), sizeof(float),
                                     cudaMemcpyDeviceToHost, stream_));

  NORDLYS_CUDA_CHECK(cudaStreamEndCapture(stream_, &graph_));
  NORDLYS_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0));

  graph_valid_ = true;
}

void CudaClusterBackend::load_centroids(const float* data, size_t n_clusters, size_t dim) {
  NVTX_RANGE_PUSH("CudaClusterBackend::load_centroids");
  
  if (n_clusters == 0 || dim == 0) {
    NVTX_RANGE_POP();
    throw std::invalid_argument("load_centroids: n_clusters and dim must be non-zero");
  }

  if (n_clusters > static_cast<size_t>(INT_MAX)) {
    NVTX_RANGE_POP();
    throw std::invalid_argument("load_centroids: n_clusters exceeds INT_MAX");
  }

  if (dim > static_cast<size_t>(INT_MAX)) {
    throw std::invalid_argument("load_centroids: dim exceeds INT_MAX");
  }

  if (data == nullptr) {
    throw std::invalid_argument("load_centroids: data pointer is null");
  }

  if (n_clusters > SIZE_MAX / dim) {
    throw std::invalid_argument("load_centroids: n_clusters * dim would overflow");
  }

  free_memory();

  n_clusters_ = static_cast<int>(n_clusters);
  dim_ = static_cast<int>(dim);

  auto nc = static_cast<size_t>(n_clusters_);
  auto d = static_cast<size_t>(dim_);

  // Use stream_ to allocate from the memory pool
  d_centroids_.reset(nc * d, stream_);      // Col-major for cuBLAS
  d_centroids_row_.reset(nc * d, stream_);  // Row-major for multi-block kernel
  d_centroid_norms_.reset(nc, stream_);
  d_embedding_.reset(d, stream_);
  d_embed_norm_.reset(1, stream_);
  d_dots_.reset(nc, stream_);
  d_best_idx_.reset(1, stream_);
  d_best_dist_.reset(1, stream_);

  h_embedding_.reset(d);
  h_best_idx_.reset(1);
  h_best_dist_.reset(1);

  auto centroids_col = cuda::to_col_major(std::span<const float>(data, n_clusters * dim), n_clusters, dim);
  auto norms = cuda::compute_squared_norms(std::span<const float>(data, n_clusters * dim), n_clusters, dim);

  // Copy col-major for cuBLAS (batch path)
  NORDLYS_CUDA_CHECK(cudaMemcpy(d_centroids_.get(), centroids_col.data(),
                                n_clusters * dim * sizeof(float), cudaMemcpyHostToDevice));
  
  // Copy row-major for multi-block kernel (single-query path) - optimal coalescing
  NORDLYS_CUDA_CHECK(cudaMemcpy(d_centroids_row_.get(), data,
                                n_clusters * dim * sizeof(float), cudaMemcpyHostToDevice));
  
  NORDLYS_CUDA_CHECK(cudaMemcpy(d_centroid_norms_.get(), norms.data(), n_clusters * sizeof(float),
                                cudaMemcpyHostToDevice));

  std::fill_n(h_embedding_.get(), dim, 1.0f);
  capture_graph();
  
  NVTX_RANGE_POP();
}

std::pair<int, float> CudaClusterBackend::assign(EmbeddingView view) {
  NVTX_RANGE_PUSH("CudaClusterBackend::assign");
  
  if (n_clusters_ == 0) {
    NVTX_RANGE_POP();
    return {-1, 0.0f};
  }

  if (view.dim != static_cast<size_t>(dim_)) {
    NVTX_RANGE_POP();
    throw std::invalid_argument("assign: dimension mismatch");
  }

  if (!graph_valid_ || !graph_exec_) {
    NVTX_RANGE_POP();
    throw std::runtime_error("CUDA graph not initialized - call load_centroids first");
  }

  auto result = std::visit(
      overloaded{[&](CpuDevice) -> std::pair<int, float> {
                   NVTX_RANGE_PUSH("assign::cpu_buffer");
                   std::memcpy(h_embedding_.get(), view.data, view.dim * sizeof(float));
                   NORDLYS_CUDA_CHECK(cudaMemcpyAsync(d_embedding_.get(), h_embedding_.get(),
                                                      view.dim * sizeof(float),
                                                      cudaMemcpyHostToDevice, stream_));

                   NORDLYS_CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream_));
                   NORDLYS_CUDA_CHECK(cudaStreamSynchronize(stream_));

                   NVTX_RANGE_POP();
                   return {*h_best_idx_.get(), *h_best_dist_.get()};
                 },
                 [&](CudaDevice d) -> std::pair<int, float> {
                   NVTX_RANGE_PUSH("assign::gpu_buffer");
                   int current_device;
                   cudaGetDevice(&current_device);
                   if (current_device != d.id) {
                     NORDLYS_CUDA_CHECK(cudaSetDevice(d.id));
                   }

                   int block_size = 256;
                   size_t shared_mem = cuda::single_query_shared_mem_size<float>(n_clusters_);
                   
                   cuda::find_nearest_centroid<float><<<1, block_size, shared_mem, stream_>>>(
                       view.data,
                       d_centroids_row_.get(),
                       d_centroid_norms_.get(),
                       d_best_idx_.get(),
                       d_best_dist_.get(),
                       n_clusters_,
                       dim_);
                   NORDLYS_CUDA_CHECK(cudaGetLastError());
                   
                   // Copy results back to host
                   NORDLYS_CUDA_CHECK(cudaMemcpyAsync(h_best_idx_.get(), d_best_idx_.get(),
                                                      sizeof(int), cudaMemcpyDeviceToHost, stream_));
                   NORDLYS_CUDA_CHECK(cudaMemcpyAsync(h_best_dist_.get(), d_best_dist_.get(),
                                                      sizeof(float), cudaMemcpyDeviceToHost, stream_));
                   NORDLYS_CUDA_CHECK(cudaStreamSynchronize(stream_));

                   NVTX_RANGE_POP();
                   return {*h_best_idx_.get(), *h_best_dist_.get()};
                 }},
      view.device);
  
  NVTX_RANGE_POP();
  return result;
}

void CudaClusterBackend::init_pipeline() {
  if (pipeline_initialized_) return;
  for (int i = 0; i < kNumPipelineStages; ++i) {
    NORDLYS_CUDA_CHECK(cudaStreamCreate(&stages_[i].stream));
    NORDLYS_CUDA_CHECK(cudaEventCreate(&stages_[i].event));
    NORDLYS_CUBLAS_CHECK(cublasCreate(&pipeline_cublas_[i]));
    NORDLYS_CUBLAS_CHECK(cublasSetStream(pipeline_cublas_[i], stages_[i].stream));
  }
  pipeline_initialized_ = true;
}

void CudaClusterBackend::ensure_stage_capacity(int stage_idx, int count) {
  auto& stage = stages_[stage_idx];
  if (count <= stage.capacity) return;

  int new_cap = count + count / 2;

  // Allocate from memory pool using the stage's stream
  stage.d_queries.reset(new_cap * dim_, stage.stream);
  stage.d_norms.reset(new_cap, stage.stream);
  stage.d_dots.reset(new_cap * n_clusters_, stage.stream);
  // Packed results: ClusterResult<float> = {int idx, float dist} = 8 bytes each
  stage.d_results.reset(new_cap * sizeof(cuda::ClusterResult<float>), stage.stream);

  stage.h_queries.reset(new_cap * dim_);
  stage.h_results.reset(new_cap * sizeof(cuda::ClusterResult<float>));

  stage.capacity = new_cap;
}

std::vector<std::pair<int, float>> CudaClusterBackend::assign_batch(EmbeddingBatchView view) {
  NVTX_RANGE_PUSH("CudaClusterBackend::assign_batch");
  
  if (view.count == 0) {
    NVTX_RANGE_POP();
    return {};
  }
  if (view.count == 1) {
    EmbeddingView single_view{view.data, view.dim, view.device};
    auto result = assign(single_view);
    NVTX_RANGE_POP();
    return {result};
  }
  if (view.dim != static_cast<size_t>(dim_)) {
    NVTX_RANGE_POP();
    throw std::invalid_argument("assign_batch: dimension mismatch");
  }

  auto result = std::visit(overloaded{[&](CpuDevice) -> std::vector<std::pair<int, float>> {
                                 NVTX_RANGE_PUSH("assign_batch::cpu_buffer");
                                 auto r = assign_batch_from_host(view);
                                 NVTX_RANGE_POP();
                                 return r;
                               },
                               [&](CudaDevice d) -> std::vector<std::pair<int, float>> {
                                 NVTX_RANGE_PUSH("assign_batch::gpu_buffer");
                                 int current_device;
                                 cudaGetDevice(&current_device);
                                 if (current_device != d.id) {
                                   NORDLYS_CUDA_CHECK(cudaSetDevice(d.id));
                                 }
                                 auto r = assign_batch_from_device(view);
                                 NVTX_RANGE_POP();
                                 return r;
                               }},
                    view.device);
  
  NVTX_RANGE_POP();
  return result;
}

std::vector<std::pair<int, float>> CudaClusterBackend::assign_batch_from_device(
    EmbeddingBatchView view) {
  init_pipeline();

  constexpr int kMinChunkSize = 64;
  int chunk_size = std::max(
      kMinChunkSize, (static_cast<int>(view.count) + kNumPipelineStages - 1) / kNumPipelineStages);
  int num_chunks = (static_cast<int>(view.count) + chunk_size - 1) / chunk_size;

  for (int i = 0; i < kNumPipelineStages; ++i) {
    ensure_stage_capacity(i, chunk_size);
  }

  std::vector<std::pair<int, float>> results(view.count);
  int block_size = 256;
  size_t shared_mem = cuda::batch_shared_mem_size<float>();
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
      auto* prev_results = reinterpret_cast<cuda::ClusterResult<float>*>(prev.h_results.get());
      for (int i = 0; i < prev_count; ++i) {
        results[prev_offset + i] = {prev_results[i].idx, prev_results[i].dist};
      }
    }

    NORDLYS_CUDA_CHECK(cudaMemcpyAsync(stage.d_queries.get(), src,
                                       this_count * dim_ * sizeof(float), cudaMemcpyDeviceToDevice,
                                       stage.stream));

    NORDLYS_CUBLAS_CHECK(cublasSgemm(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, n_clusters_, this_count, dim_, &alpha, d_centroids_.get(),
        n_clusters_, stage.d_queries.get(), dim_, &beta, stage.d_dots.get(), n_clusters_));

    auto* d_results = reinterpret_cast<cuda::ClusterResult<float>*>(stage.d_results.get());
    cuda::find_nearest_centroid_batch<<<this_count, block_size, shared_mem, stage.stream>>>(
        stage.d_queries.get(), d_centroid_norms_.get(), stage.d_dots.get(), this_count, n_clusters_,
        dim_, d_results);

    // Single coalesced memcpy instead of two separate transfers
    NORDLYS_CUDA_CHECK(cudaMemcpyAsync(stage.h_results.get(), stage.d_results.get(),
                                       this_count * sizeof(cuda::ClusterResult<float>),
                                       cudaMemcpyDeviceToHost, stage.stream));

    NORDLYS_CUDA_CHECK(cudaEventRecord(stage.event, stage.stream));
  }

  for (int tail = std::max(0, num_chunks - kNumPipelineStages); tail < num_chunks; ++tail) {
    int stage_idx = tail % kNumPipelineStages;
    auto& stage = stages_[stage_idx];

    NORDLYS_CUDA_CHECK(cudaEventSynchronize(stage.event));

    int offset = tail * chunk_size;
    int this_count = std::min(chunk_size, static_cast<int>(view.count) - offset);
    auto* h_results = reinterpret_cast<cuda::ClusterResult<float>*>(stage.h_results.get());
    for (int i = 0; i < this_count; ++i) {
      results[offset + i] = {h_results[i].idx, h_results[i].dist};
    }
  }

  return results;
}

std::vector<std::pair<int, float>> CudaClusterBackend::assign_batch_from_host(
    EmbeddingBatchView view) {
  init_pipeline();

  constexpr int kMinChunkSize = 64;
  int chunk_size = std::max(
      kMinChunkSize, (static_cast<int>(view.count) + kNumPipelineStages - 1) / kNumPipelineStages);
  int num_chunks = (static_cast<int>(view.count) + chunk_size - 1) / chunk_size;

  for (int i = 0; i < kNumPipelineStages; ++i) {
    ensure_stage_capacity(i, chunk_size);
  }

  std::vector<std::pair<int, float>> results(view.count);
  int block_size = 256;
  size_t shared_mem = cuda::batch_shared_mem_size<float>();
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
      auto* prev_results = reinterpret_cast<cuda::ClusterResult<float>*>(prev.h_results.get());
      for (int i = 0; i < prev_count; ++i) {
        results[prev_offset + i] = {prev_results[i].idx, prev_results[i].dist};
      }
    }

    NORDLYS_CUDA_CHECK(cudaMemcpyAsync(stage.d_queries.get(), src,
                                       this_count * dim_ * sizeof(float), cudaMemcpyHostToDevice,
                                       stage.stream));

    NORDLYS_CUBLAS_CHECK(cublasSgemm(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, n_clusters_, this_count, dim_, &alpha, d_centroids_.get(),
        n_clusters_, stage.d_queries.get(), dim_, &beta, stage.d_dots.get(), n_clusters_));

    auto* d_results = reinterpret_cast<cuda::ClusterResult<float>*>(stage.d_results.get());
    cuda::find_nearest_centroid_batch<<<this_count, block_size, shared_mem, stage.stream>>>(
        stage.d_queries.get(), d_centroid_norms_.get(), stage.d_dots.get(), this_count, n_clusters_,
        dim_, d_results);

    // Single coalesced memcpy instead of two separate transfers
    NORDLYS_CUDA_CHECK(cudaMemcpyAsync(stage.h_results.get(), stage.d_results.get(),
                                       this_count * sizeof(cuda::ClusterResult<float>),
                                       cudaMemcpyDeviceToHost, stage.stream));

    NORDLYS_CUDA_CHECK(cudaEventRecord(stage.event, stage.stream));
  }

  for (int tail = std::max(0, num_chunks - kNumPipelineStages); tail < num_chunks; ++tail) {
    int stage_idx = tail % kNumPipelineStages;
    auto& stage = stages_[stage_idx];

    NORDLYS_CUDA_CHECK(cudaEventSynchronize(stage.event));

    int offset = tail * chunk_size;
    int this_count = std::min(chunk_size, static_cast<int>(view.count) - offset);
    auto* h_results = reinterpret_cast<cuda::ClusterResult<float>*>(stage.h_results.get());
    for (int i = 0; i < this_count; ++i) {
      results[offset + i] = {h_results[i].idx, h_results[i].dist};
    }
  }

  return results;
}

// =============================================================================
// CUDA utility function
// =============================================================================

bool cuda_available() noexcept {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  return err == cudaSuccess && device_count > 0;
}

}  // namespace nordlys::clustering

#endif  // NORDLYS_HAS_CUDA
