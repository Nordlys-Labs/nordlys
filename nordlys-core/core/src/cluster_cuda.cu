#ifdef NORDLYS_HAS_CUDA

#include <nordlys_core/cluster.hpp>
#include <nordlys_core/distance_cuda.cuh>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do { \
  cudaError_t err = (call); \
  if (err != cudaSuccess) { \
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
  } \
} while (0)

#define CUBLAS_CHECK(call) do { \
  cublasStatus_t stat = (call); \
  if (stat != CUBLAS_STATUS_SUCCESS) { \
    throw std::runtime_error(std::string("cuBLAS error: ") + std::to_string(stat)); \
  } \
} while (0)

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

using namespace nordlys::cuda;

// Float specialization
template <>
void CudaClusterBackend<float>::free_memory() {
  if (graph_exec_) { cudaGraphExecDestroy(graph_exec_); graph_exec_ = nullptr; }
  if (graph_) { cudaGraphDestroy(graph_); graph_ = nullptr; }
  if (d_centroids_) { cudaFree(d_centroids_); d_centroids_ = nullptr; }
  if (d_centroid_norms_) { cudaFree(d_centroid_norms_); d_centroid_norms_ = nullptr; }
  if (d_embedding_) { cudaFree(d_embedding_); d_embedding_ = nullptr; }
  if (d_embed_norm_) { cudaFree(d_embed_norm_); d_embed_norm_ = nullptr; }
  if (d_dots_) { cudaFree(d_dots_); d_dots_ = nullptr; }
  if (d_best_idx_) { cudaFree(d_best_idx_); d_best_idx_ = nullptr; }
  if (d_best_dist_) { cudaFree(d_best_dist_); d_best_dist_ = nullptr; }
  if (h_embedding_) { cudaFreeHost(h_embedding_); h_embedding_ = nullptr; }
  if (h_best_idx_) { cudaFreeHost(h_best_idx_); h_best_idx_ = nullptr; }
  if (h_best_dist_) { cudaFreeHost(h_best_dist_); h_best_dist_ = nullptr; }
}

template <>
CudaClusterBackend<float>::CudaClusterBackend() {
  try {
    CUDA_CHECK(cudaStreamCreate(&stream_));
    CUBLAS_CHECK(cublasCreate(&cublas_));
    CUBLAS_CHECK(cublasSetStream(cublas_, stream_));
  } catch (...) {
    // Destructor will clean up any successfully created resources
    // since members are initialized to nullptr in the header
    if (cublas_) { cublasDestroy(cublas_); cublas_ = nullptr; }
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    throw;
  }
}

template <>
CudaClusterBackend<float>::~CudaClusterBackend() {
  free_memory();
  if (cublas_) { cublasDestroy(cublas_); }
  if (stream_) { cudaStreamDestroy(stream_); }
}

template <>
void CudaClusterBackend<float>::capture_graph() {
  if (graph_exec_) { cudaGraphExecDestroy(graph_exec_); graph_exec_ = nullptr; }
  if (graph_) { cudaGraphDestroy(graph_); graph_ = nullptr; }

  CUDA_CHECK(cudaStreamSynchronize(stream_));
  CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));

  // H2D: copy embedding only (norm computed on GPU)
  CUDA_CHECK(cudaMemcpyAsync(d_embedding_, h_embedding_, 
                             dim_ * sizeof(float), cudaMemcpyHostToDevice, stream_));

  // Compute embedding norm on GPU (removes one H2D transfer)
  // Use smaller block for small dims to reduce latency
  int norm_block = (dim_ <= 512) ? 128 : 256;
  compute_squared_norm<float><<<1, norm_block, 0, stream_>>>(
      d_embedding_, dim_, d_embed_norm_);

  // cuBLAS GEMV: dot products
  float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasSgemv(cublas_, CUBLAS_OP_T, dim_, n_clusters_,
                           &alpha, d_centroids_, dim_,
                           d_embedding_, 1,
                           &beta, d_dots_, 1));

  // Argmin kernel - use 64 threads for small cluster counts (better for 20-100 clusters)
  int argmin_block = (n_clusters_ <= 128) ? 64 : 128;
  fused_l2_argmin<float><<<1, argmin_block, 0, stream_>>>(
      d_centroid_norms_, d_embed_norm_, d_dots_, n_clusters_,
      d_best_idx_, d_best_dist_);
  CUDA_CHECK(cudaGetLastError());

  // D2H: copy results
  CUDA_CHECK(cudaMemcpyAsync(h_best_idx_, d_best_idx_, sizeof(int),
                             cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaMemcpyAsync(h_best_dist_, d_best_dist_, sizeof(float),
                             cudaMemcpyDeviceToHost, stream_));

  CUDA_CHECK(cudaStreamEndCapture(stream_, &graph_));
  CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0));
  
  graph_valid_ = true;
}

template <>
void CudaClusterBackend<float>::load_centroids(const float* data, int n_clusters, int dim) {
  if (n_clusters <= 0 || dim <= 0) {
    throw std::invalid_argument("n_clusters and dim must be positive");
  }

  free_memory();

  n_clusters_ = n_clusters;
  dim_ = dim;
  auto nc = static_cast<size_t>(n_clusters);
  auto d = static_cast<size_t>(dim);

  // Device memory
  CUDA_CHECK(cudaMalloc(&d_centroids_, nc * d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_centroid_norms_, nc * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_embedding_, d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_embed_norm_, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dots_, nc * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_best_idx_, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_best_dist_, sizeof(float)));

  // Pinned host memory
  CUDA_CHECK(cudaMallocHost(&h_embedding_, d * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&h_best_idx_, sizeof(int)));
  CUDA_CHECK(cudaMallocHost(&h_best_dist_, sizeof(float)));

  // Upload centroids
  auto centroids_col = to_col_major(data, nc, d);
  auto norms = compute_squared_norms_cpu(data, nc, d);

  CUDA_CHECK(cudaMemcpy(d_centroids_, centroids_col.data(), nc * d * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_centroid_norms_, norms.data(), nc * sizeof(float), cudaMemcpyHostToDevice));

  // Capture CUDA graph with dummy data
  std::fill_n(h_embedding_, d, 1.0f);
  capture_graph();
}

template <>
std::pair<int, float> CudaClusterBackend<float>::assign(const float* embedding, int dim) {
  if (n_clusters_ == 0 || dim != dim_) {
    return {-1, 0.0f};
  }

  if (!graph_valid_ || !graph_exec_) {
    throw std::runtime_error("CUDA graph not initialized - call load_centroids first");
  }

  // Copy to pinned memory (norm computed on GPU)
  std::memcpy(h_embedding_, embedding, static_cast<size_t>(dim) * sizeof(float));

  // Launch graph (replays entire pipeline)
  CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));

  return {*h_best_idx_, *h_best_dist_};
}

// Double specialization
template <>
void CudaClusterBackend<double>::free_memory() {
  if (graph_exec_) { cudaGraphExecDestroy(graph_exec_); graph_exec_ = nullptr; }
  if (graph_) { cudaGraphDestroy(graph_); graph_ = nullptr; }
  if (d_centroids_) { cudaFree(d_centroids_); d_centroids_ = nullptr; }
  if (d_centroid_norms_) { cudaFree(d_centroid_norms_); d_centroid_norms_ = nullptr; }
  if (d_embedding_) { cudaFree(d_embedding_); d_embedding_ = nullptr; }
  if (d_embed_norm_) { cudaFree(d_embed_norm_); d_embed_norm_ = nullptr; }
  if (d_dots_) { cudaFree(d_dots_); d_dots_ = nullptr; }
  if (d_best_idx_) { cudaFree(d_best_idx_); d_best_idx_ = nullptr; }
  if (d_best_dist_) { cudaFree(d_best_dist_); d_best_dist_ = nullptr; }
  if (h_embedding_) { cudaFreeHost(h_embedding_); h_embedding_ = nullptr; }
  if (h_best_idx_) { cudaFreeHost(h_best_idx_); h_best_idx_ = nullptr; }
  if (h_best_dist_) { cudaFreeHost(h_best_dist_); h_best_dist_ = nullptr; }
}

template <>
CudaClusterBackend<double>::CudaClusterBackend() {
  try {
    CUDA_CHECK(cudaStreamCreate(&stream_));
    CUBLAS_CHECK(cublasCreate(&cublas_));
    CUBLAS_CHECK(cublasSetStream(cublas_, stream_));
  } catch (...) {
    // Clean up any successfully created resources before re-throwing
    if (cublas_) { cublasDestroy(cublas_); cublas_ = nullptr; }
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    throw;
  }
}

template <>
CudaClusterBackend<double>::~CudaClusterBackend() {
  free_memory();
  if (cublas_) { cublasDestroy(cublas_); }
  if (stream_) { cudaStreamDestroy(stream_); }
}

template <>
void CudaClusterBackend<double>::capture_graph() {
  if (graph_exec_) { cudaGraphExecDestroy(graph_exec_); graph_exec_ = nullptr; }
  if (graph_) { cudaGraphDestroy(graph_); graph_ = nullptr; }

  CUDA_CHECK(cudaStreamSynchronize(stream_));
  CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));

  CUDA_CHECK(cudaMemcpyAsync(d_embedding_, h_embedding_, 
                             dim_ * sizeof(double), cudaMemcpyHostToDevice, stream_));

  int norm_block = (dim_ <= 512) ? 128 : 256;
  compute_squared_norm<double><<<1, norm_block, 0, stream_>>>(
      d_embedding_, dim_, d_embed_norm_);

  double alpha = 1.0, beta = 0.0;
  CUBLAS_CHECK(cublasDgemv(cublas_, CUBLAS_OP_T, dim_, n_clusters_,
                           &alpha, d_centroids_, dim_,
                           d_embedding_, 1,
                           &beta, d_dots_, 1));

  int argmin_block = (n_clusters_ <= 128) ? 64 : 128;
  fused_l2_argmin<double><<<1, argmin_block, 0, stream_>>>(
      d_centroid_norms_, d_embed_norm_, d_dots_, n_clusters_,
      d_best_idx_, d_best_dist_);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpyAsync(h_best_idx_, d_best_idx_, sizeof(int),
                             cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaMemcpyAsync(h_best_dist_, d_best_dist_, sizeof(double),
                             cudaMemcpyDeviceToHost, stream_));

  CUDA_CHECK(cudaStreamEndCapture(stream_, &graph_));
  CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0));
  
  graph_valid_ = true;
}

template <>
void CudaClusterBackend<double>::load_centroids(const double* data, int n_clusters, int dim) {
  if (n_clusters <= 0 || dim <= 0) {
    throw std::invalid_argument("n_clusters and dim must be positive");
  }

  free_memory();

  n_clusters_ = n_clusters;
  dim_ = dim;
  auto nc = static_cast<size_t>(n_clusters);
  auto d = static_cast<size_t>(dim);

  CUDA_CHECK(cudaMalloc(&d_centroids_, nc * d * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_centroid_norms_, nc * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_embedding_, d * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_embed_norm_, sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_dots_, nc * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_best_idx_, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_best_dist_, sizeof(double)));
  CUDA_CHECK(cudaMallocHost(&h_embedding_, d * sizeof(double)));
  CUDA_CHECK(cudaMallocHost(&h_best_idx_, sizeof(int)));
  CUDA_CHECK(cudaMallocHost(&h_best_dist_, sizeof(double)));

  auto centroids_col = to_col_major(data, nc, d);
  auto norms = compute_squared_norms_cpu(data, nc, d);

  CUDA_CHECK(cudaMemcpy(d_centroids_, centroids_col.data(), nc * d * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_centroid_norms_, norms.data(), nc * sizeof(double), cudaMemcpyHostToDevice));

  std::fill_n(h_embedding_, d, 1.0);
  capture_graph();
}

template <>
std::pair<int, double> CudaClusterBackend<double>::assign(const double* embedding, int dim) {
  if (n_clusters_ == 0 || dim != dim_) {
    return {-1, 0.0};
  }

  if (!graph_valid_ || !graph_exec_) {
    throw std::runtime_error("CUDA graph not initialized - call load_centroids first");
  }

  std::memcpy(h_embedding_, embedding, static_cast<size_t>(dim) * sizeof(double));

  CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));

  return {*h_best_idx_, *h_best_dist_};
}

#endif  // NORDLYS_HAS_CUDA
