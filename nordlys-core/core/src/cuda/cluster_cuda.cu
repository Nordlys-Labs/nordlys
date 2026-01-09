#ifdef NORDLYS_HAS_CUDA

#include <nordlys_core/cuda/cluster_cuda.hpp>

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

template <typename Scalar>
std::vector<Scalar> compute_norms(const Scalar* data, size_t n_clusters, size_t dim) {
  std::vector<Scalar> norms(n_clusters);
  for (size_t k = 0; k < n_clusters; ++k) {
    const Scalar* row = data + k * dim;
    norms[k] = std::inner_product(row, row + dim, row, Scalar{0});
  }
  return norms;
}

template<typename Scalar>
std::vector<Scalar> to_col_major(const Scalar* data, size_t n_clusters, size_t dim) {
  return std::vector<Scalar>(data, data + n_clusters * dim);
}

}  // namespace

template <typename Scalar>
__device__ __forceinline__ Scalar warp_reduce_sum(Scalar val) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// Optimized norm kernel with vectorized loads for float
template <typename Scalar>
__global__ void compute_norm_kernel(
    const Scalar* __restrict__ embedding,
    int dim,
    Scalar* __restrict__ norm_out) {
  
  __shared__ Scalar s_partial[32];
  
  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp_id = tid >> 5;
  
  Scalar sum = Scalar(0);
  
  // Process 4 elements at a time for better memory throughput
  int vec_dim = (dim / 4) * 4;
  for (int i = tid * 4; i < vec_dim; i += blockDim.x * 4) {
    if (i + 3 < dim) {
      Scalar v0 = embedding[i];
      Scalar v1 = embedding[i + 1];
      Scalar v2 = embedding[i + 2];
      Scalar v3 = embedding[i + 3];
      sum += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
    }
  }
  
  // Handle remaining elements
  for (int i = vec_dim + tid; i < dim; i += blockDim.x) {
    Scalar val = embedding[i];
    sum += val * val;
  }
  
  sum = warp_reduce_sum(sum);
  
  if (lane == 0) {
    s_partial[warp_id] = sum;
  }
  __syncthreads();
  
  if (warp_id == 0) {
    int num_warps = (blockDim.x + 31) >> 5;
    sum = (lane < num_warps) ? s_partial[lane] : Scalar(0);
    sum = warp_reduce_sum(sum);
    if (lane == 0) {
      *norm_out = sum;
    }
  }
}

// Optimized argmin with vectorized loads and ILP
template <typename Scalar>
__global__ void fused_argmin_kernel(
    const Scalar* __restrict__ c_norms,
    const Scalar* __restrict__ e_norm_ptr,
    const Scalar* __restrict__ dots,
    int n,
    int* __restrict__ best_idx,
    Scalar* __restrict__ best_dist) {

  __shared__ Scalar s_min_dist[32];
  __shared__ int s_min_idx[32];

  Scalar e_norm = *e_norm_ptr;
  
  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp_id = tid >> 5;

  Scalar local_min = INFINITY;
  int local_idx = -1;

  // Process 4 elements per iteration for better ILP
  // Each thread handles indices: tid*4, tid*4 + blockDim.x*4, tid*4 + blockDim.x*8, ...
  const Scalar two = Scalar(2);
  int vec_end = (n / 4) * 4;
  for (int i = tid * 4; i < vec_end; i += blockDim.x * 4) {
    Scalar d0 = c_norms[i] + e_norm - two * dots[i];
    Scalar d1 = c_norms[i + 1] + e_norm - two * dots[i + 1];
    Scalar d2 = c_norms[i + 2] + e_norm - two * dots[i + 2];
    Scalar d3 = c_norms[i + 3] + e_norm - two * dots[i + 3];
    
    if (d0 < local_min) { local_min = d0; local_idx = i; }
    if (d1 < local_min) { local_min = d1; local_idx = i + 1; }
    if (d2 < local_min) { local_min = d2; local_idx = i + 2; }
    if (d3 < local_min) { local_min = d3; local_idx = i + 3; }
  }
  
  // Handle remaining elements (n % 4 elements starting at vec_end)
  for (int j = vec_end + tid; j < n; j += blockDim.x) {
    Scalar dist = c_norms[j] + e_norm - two * dots[j];
    if (dist < local_min) {
      local_min = dist;
      local_idx = j;
    }
  }

  // Warp reduction
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    Scalar other_dist = __shfl_down_sync(0xffffffff, local_min, offset);
    int other_idx = __shfl_down_sync(0xffffffff, local_idx, offset);
    if (other_dist < local_min) {
      local_min = other_dist;
      local_idx = other_idx;
    }
  }

  if (lane == 0) {
    s_min_dist[warp_id] = local_min;
    s_min_idx[warp_id] = local_idx;
  }
  __syncthreads();

  // Final reduction
  if (warp_id == 0) {
    int num_warps = (blockDim.x + 31) >> 5;
    local_min = (lane < num_warps) ? s_min_dist[lane] : INFINITY;
    local_idx = (lane < num_warps) ? s_min_idx[lane] : -1;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      Scalar other_dist = __shfl_down_sync(0xffffffff, local_min, offset);
      int other_idx = __shfl_down_sync(0xffffffff, local_idx, offset);
      if (other_dist < local_min) {
        local_min = other_dist;
        local_idx = other_idx;
      }
    }

    if (lane == 0) {
      *best_idx = local_idx;
      *best_dist = sqrt(max(local_min, Scalar(0)));
    }
  }
}

// Float specialization
template <>
void CudaClusterBackendT<float>::free_memory() {
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
CudaClusterBackendT<float>::CudaClusterBackendT() {
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
CudaClusterBackendT<float>::~CudaClusterBackendT() {
  free_memory();
  if (cublas_) { cublasDestroy(cublas_); }
  if (stream_) { cudaStreamDestroy(stream_); }
}

template <>
void CudaClusterBackendT<float>::capture_graph() {
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
  compute_norm_kernel<float><<<1, norm_block, 0, stream_>>>(
      d_embedding_, dim_, d_embed_norm_);

  // cuBLAS GEMV: dot products
  float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasSgemv(cublas_, CUBLAS_OP_T, dim_, n_clusters_,
                           &alpha, d_centroids_, dim_,
                           d_embedding_, 1,
                           &beta, d_dots_, 1));

  // Argmin kernel - use 64 threads for small cluster counts (better for 20-100 clusters)
  int argmin_block = (n_clusters_ <= 128) ? 64 : 128;
  fused_argmin_kernel<float><<<1, argmin_block, 0, stream_>>>(
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
void CudaClusterBackendT<float>::load_centroids(const float* data, int n_clusters, int dim) {
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
  auto norms = compute_norms(data, nc, d);

  CUDA_CHECK(cudaMemcpy(d_centroids_, centroids_col.data(), nc * d * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_centroid_norms_, norms.data(), nc * sizeof(float), cudaMemcpyHostToDevice));

  // Capture CUDA graph with dummy data
  std::fill_n(h_embedding_, d, 1.0f);
  capture_graph();
}

template <>
std::pair<int, float> CudaClusterBackendT<float>::assign(const float* embedding, int dim) {
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
void CudaClusterBackendT<double>::free_memory() {
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
CudaClusterBackendT<double>::CudaClusterBackendT() {
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
CudaClusterBackendT<double>::~CudaClusterBackendT() {
  free_memory();
  if (cublas_) { cublasDestroy(cublas_); }
  if (stream_) { cudaStreamDestroy(stream_); }
}

template <>
void CudaClusterBackendT<double>::capture_graph() {
  if (graph_exec_) { cudaGraphExecDestroy(graph_exec_); graph_exec_ = nullptr; }
  if (graph_) { cudaGraphDestroy(graph_); graph_ = nullptr; }

  CUDA_CHECK(cudaStreamSynchronize(stream_));
  CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));

  CUDA_CHECK(cudaMemcpyAsync(d_embedding_, h_embedding_, 
                             dim_ * sizeof(double), cudaMemcpyHostToDevice, stream_));

  int norm_block = (dim_ <= 512) ? 128 : 256;
  compute_norm_kernel<double><<<1, norm_block, 0, stream_>>>(
      d_embedding_, dim_, d_embed_norm_);

  double alpha = 1.0, beta = 0.0;
  CUBLAS_CHECK(cublasDgemv(cublas_, CUBLAS_OP_T, dim_, n_clusters_,
                           &alpha, d_centroids_, dim_,
                           d_embedding_, 1,
                           &beta, d_dots_, 1));

  int argmin_block = (n_clusters_ <= 128) ? 64 : 128;
  fused_argmin_kernel<double><<<1, argmin_block, 0, stream_>>>(
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
void CudaClusterBackendT<double>::load_centroids(const double* data, int n_clusters, int dim) {
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
  auto norms = compute_norms(data, nc, d);

  CUDA_CHECK(cudaMemcpy(d_centroids_, centroids_col.data(), nc * d * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_centroid_norms_, norms.data(), nc * sizeof(double), cudaMemcpyHostToDevice));

  std::fill_n(h_embedding_, d, 1.0);
  capture_graph();
}

template <>
std::pair<int, double> CudaClusterBackendT<double>::assign(const double* embedding, int dim) {
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
