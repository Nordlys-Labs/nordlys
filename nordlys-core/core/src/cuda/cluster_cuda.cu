#ifdef NORDLYS_HAS_CUDA

#include <nordlys_core/cuda/cluster_cuda.hpp>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template<typename Scalar>
std::vector<Scalar> to_col_major(const Scalar* data, size_t n_clusters, size_t dim) {
  // Row-major K×D is mathematically equivalent to column-major D×K when interpreted as transpose
  // No conversion needed - just copy the data
  return std::vector<Scalar>(data, data + n_clusters * dim);
}

template<typename Scalar>
std::vector<Scalar> compute_norms(const Scalar* data, size_t n_clusters, size_t dim) {
  std::vector<Scalar> norms(n_clusters);
  std::ranges::transform(std::views::iota(size_t{0}, n_clusters), norms.begin(),
                         [&](size_t k) {
                           const Scalar* row = data + k * dim;
                           return std::inner_product(row, row + dim, row, static_cast<Scalar>(0));
                         });
  return norms;
}

}  // namespace

// CUDA error checking macro
#define CUDA_CHECK(call)                                                                           \
  do {                                                                                             \
    cudaError_t err = (call);                                                                      \
    if (err != cudaSuccess) {                                                                      \
      throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));             \
    }                                                                                              \
  } while (0)

// cuBLAS error checking macro
#define CUBLAS_CHECK(call)                                                                         \
  do {                                                                                             \
    cublasStatus_t stat = (call);                                                                  \
    if (stat != CUBLAS_STATUS_SUCCESS) {                                                           \
      throw std::runtime_error(std::string("cuBLAS error: ") + std::to_string(stat));             \
    }                                                                                              \
  } while (0)

// Fused kernel: Compute distances and find argmin using warp reduction
// Much more efficient than separate kernel + Thrust for small n_clusters (20-50)
template<typename Scalar>
__global__ void fused_distance_argmin_kernel(
    const Scalar* __restrict__ c_norms,
    Scalar e_norm,
    const Scalar* __restrict__ dots,
    int n,
    int* __restrict__ best_idx,
    Scalar* __restrict__ best_dist) {

  // Shared memory for warp reduction (max 32 warps per block)
  __shared__ Scalar s_min_dist[32];
  __shared__ int s_min_idx[32];

  int tid = threadIdx.x;
  int lane = tid & 31;  // Lane within warp
  int warp_id = tid >> 5;  // Warp ID within block

  Scalar local_min = INFINITY;
  int local_idx = -1;

  // Grid-stride loop for robustness with large n_clusters
  for (int i = tid; i < n; i += blockDim.x) {
    // Use FMA for better precision and potential performance: dist = ||c_i||² + ||e||² - 2*dot[i]
    Scalar dist = fma(Scalar(-2.0), dots[i], c_norms[i] + e_norm);
    if (dist < local_min) {
      local_min = dist;
      local_idx = i;
    }
  }

  // Warp reduction using shuffle operations
  for (int offset = 16; offset > 0; offset >>= 1) {
    Scalar other_dist = __shfl_down_sync(0xffffffff, local_min, offset);
    int other_idx = __shfl_down_sync(0xffffffff, local_idx, offset);
    if (other_dist < local_min) {
      local_min = other_dist;
      local_idx = other_idx;
    }
  }

  // First lane of each warp writes to shared memory
  if (lane == 0) {
    s_min_dist[warp_id] = local_min;
    s_min_idx[warp_id] = local_idx;
  }
  __syncthreads();

  // First warp reduces across warps
  if (warp_id == 0) {
    int num_warps = (blockDim.x + 31) >> 5;
    local_min = (lane < num_warps) ? s_min_dist[lane] : INFINITY;
    local_idx = (lane < num_warps) ? s_min_idx[lane] : -1;

    for (int offset = 16; offset > 0; offset >>= 1) {
      Scalar other_dist = __shfl_down_sync(0xffffffff, local_min, offset);
      int other_idx = __shfl_down_sync(0xffffffff, local_idx, offset);
      if (other_dist < local_min) {
        local_min = other_dist;
        local_idx = other_idx;
      }
    }

    // First thread writes final result
    if (lane == 0) {
      *best_idx = local_idx;
      *best_dist = sqrt(max(local_min, Scalar(0)));
    }
  }
}

// Template specialization for float - free_device_memory must come first (called by destructor)
template<>
void CudaClusterBackendT<float>::free_device_memory() {
  if (d_centroids_) {
    cudaFree(d_centroids_);
    d_centroids_ = nullptr;
  }
  if (d_centroid_norms_) {
    cudaFree(d_centroid_norms_);
    d_centroid_norms_ = nullptr;
  }
  if (d_embedding_) {
    cudaFree(d_embedding_);
    d_embedding_ = nullptr;
  }
  if (d_dots_) {
    cudaFree(d_dots_);
    d_dots_ = nullptr;
  }
  if (d_best_idx_) {
    cudaFree(d_best_idx_);
    d_best_idx_ = nullptr;
  }
  if (d_best_dist_) {
    cudaFree(d_best_dist_);
    d_best_dist_ = nullptr;
  }
}

template<>
CudaClusterBackendT<float>::CudaClusterBackendT() {
  CUDA_CHECK(cudaStreamCreate(&stream_));
  CUBLAS_CHECK(cublasCreate(&cublas_handle_));
  CUBLAS_CHECK(cublasSetStream(cublas_handle_, stream_));
}

template<>
CudaClusterBackendT<float>::~CudaClusterBackendT() {
  free_device_memory();
  if (cublas_handle_) {
    cublasDestroy(cublas_handle_);
  }
  if (stream_) {
    cudaStreamDestroy(stream_);
  }
}

template<>
void CudaClusterBackendT<float>::load_centroids(const float* data, int n_clusters, int dim) {
  if (n_clusters <= 0 || dim <= 0) {
    throw std::invalid_argument("n_clusters and dim must be positive");
  }

  free_device_memory();

  n_clusters_ = n_clusters;
  dim_ = dim;
  const size_t n_clusters_sz = static_cast<size_t>(n_clusters_);
  const size_t dim_sz = static_cast<size_t>(dim_);

  // Allocate device memory
  const size_t centroids_size = n_clusters_sz * dim_sz * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_centroids_, centroids_size));
  CUDA_CHECK(cudaMalloc(&d_centroid_norms_, n_clusters_sz * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_embedding_, dim_sz * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dots_, n_clusters_sz * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_best_idx_, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_best_dist_, sizeof(float)));

   // Row-major [K][D] is mathematically equivalent to column-major [D][K] when interpreted as transpose
   std::vector<float> centroids_col_major = to_col_major(data, n_clusters_sz, dim_sz);

  // Compute centroid norms: ||c_i||²
  std::vector<float> centroid_norms = compute_norms(data, n_clusters_sz, dim_sz);

  // Upload to device
  CUDA_CHECK(cudaMemcpy(d_centroids_, centroids_col_major.data(), centroids_size,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_centroid_norms_, centroid_norms.data(),
                        n_clusters_sz * sizeof(float), cudaMemcpyHostToDevice));
}

template<>
std::pair<int, float> CudaClusterBackendT<float>::assign(const float* embedding, int dim) {
  if (n_clusters_ == 0 || dim != dim_) {
    return {-1, 0.0f};
  }

  // 1. Compute embedding norm on host
  const size_t dim_sz = static_cast<size_t>(dim);
  const float embed_norm =
      std::inner_product(embedding, embedding + dim_sz, embedding, static_cast<float>(0));

  // 2. Copy embedding to device
  CUDA_CHECK(cudaMemcpyAsync(d_embedding_, embedding, static_cast<size_t>(dim) * sizeof(float),
                             cudaMemcpyHostToDevice, stream_));

  // 3. cuBLAS SGEMV: dot[i] = centroids^T @ embedding
  // centroids is [dim x n_clusters] in column-major
  // embedding is [dim x 1]
  // result is [n_clusters x 1]
  float alpha = 1.0f;
  float beta = 0.0f;
  CUBLAS_CHECK(cublasSgemv(cublas_handle_, CUBLAS_OP_T,  // transpose centroids
                           dim_, n_clusters_,             // m, n
                           &alpha, d_centroids_, dim_,    // A, lda
                           d_embedding_, 1,               // x, incx
                           &beta, d_dots_, 1));           // y, incy

  // 4. Fused kernel: compute distances and find argmin in single kernel
  constexpr int kBlockSize = 256;  // Good for small n_clusters (20-50)
  fused_distance_argmin_kernel<float><<<1, kBlockSize, 0, stream_>>>(
      d_centroid_norms_, embed_norm, d_dots_, n_clusters_,
      d_best_idx_, d_best_dist_);

  // Check for kernel launch errors
  CUDA_CHECK(cudaGetLastError());

  // 5. Copy results back to host
  int host_best_idx;
  float host_best_dist;
  CUDA_CHECK(cudaMemcpyAsync(&host_best_idx, d_best_idx_, sizeof(int),
                            cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaMemcpyAsync(&host_best_dist, d_best_dist_, sizeof(float),
                            cudaMemcpyDeviceToHost, stream_));

  // 6. Synchronize to ensure copies complete
  CUDA_CHECK(cudaStreamSynchronize(stream_));

  return {host_best_idx, host_best_dist};
}

// Template specialization for double - free_device_memory must come first (called by destructor)
template<>
void CudaClusterBackendT<double>::free_device_memory() {
  if (d_centroids_) {
    cudaFree(d_centroids_);
    d_centroids_ = nullptr;
  }
  if (d_centroid_norms_) {
    cudaFree(d_centroid_norms_);
    d_centroid_norms_ = nullptr;
  }
  if (d_embedding_) {
    cudaFree(d_embedding_);
    d_embedding_ = nullptr;
  }
  if (d_dots_) {
    cudaFree(d_dots_);
    d_dots_ = nullptr;
  }
  if (d_best_idx_) {
    cudaFree(d_best_idx_);
    d_best_idx_ = nullptr;
  }
  if (d_best_dist_) {
    cudaFree(d_best_dist_);
    d_best_dist_ = nullptr;
  }
}

template<>
CudaClusterBackendT<double>::CudaClusterBackendT() {
  CUDA_CHECK(cudaStreamCreate(&stream_));
  CUBLAS_CHECK(cublasCreate(&cublas_handle_));
  CUBLAS_CHECK(cublasSetStream(cublas_handle_, stream_));
}

template<>
CudaClusterBackendT<double>::~CudaClusterBackendT() {
  free_device_memory();
  if (cublas_handle_) {
    cublasDestroy(cublas_handle_);
  }
  if (stream_) {
    cudaStreamDestroy(stream_);
  }
}

template<>
void CudaClusterBackendT<double>::load_centroids(const double* data, int n_clusters, int dim) {
  if (n_clusters <= 0 || dim <= 0) {
    throw std::invalid_argument("n_clusters and dim must be positive");
  }

  free_device_memory();

  n_clusters_ = n_clusters;
  dim_ = dim;
  const size_t n_clusters_sz = static_cast<size_t>(n_clusters_);
  const size_t dim_sz = static_cast<size_t>(dim_);

  // Allocate device memory
  const size_t centroids_size = n_clusters_sz * dim_sz * sizeof(double);
  CUDA_CHECK(cudaMalloc(&d_centroids_, centroids_size));
  CUDA_CHECK(cudaMalloc(&d_centroid_norms_, n_clusters_sz * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_embedding_, dim_sz * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_dots_, n_clusters_sz * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_best_idx_, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_best_dist_, sizeof(double)));

   // Row-major [K][D] is mathematically equivalent to column-major [D][K] when interpreted as transpose
   std::vector<double> centroids_col_major = to_col_major(data, n_clusters_sz, dim_sz);

  // Compute centroid norms: ||c_i||²
  std::vector<double> centroid_norms = compute_norms(data, n_clusters_sz, dim_sz);

  // Upload to device
  CUDA_CHECK(cudaMemcpy(d_centroids_, centroids_col_major.data(), centroids_size,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_centroid_norms_, centroid_norms.data(),
                        static_cast<size_t>(n_clusters) * sizeof(double),
                        cudaMemcpyHostToDevice));
}

template<>
std::pair<int, double> CudaClusterBackendT<double>::assign(const double* embedding, int dim) {
  if (n_clusters_ == 0 || dim != dim_) {
    return {-1, 0.0};
  }

  // 1. Compute embedding norm on host
  const size_t dim_sz = static_cast<size_t>(dim);
  const double embed_norm =
      std::inner_product(embedding, embedding + dim_sz, embedding, static_cast<double>(0));

  // 2. Copy embedding to device
  CUDA_CHECK(cudaMemcpyAsync(d_embedding_, embedding, static_cast<size_t>(dim) * sizeof(double),
                             cudaMemcpyHostToDevice, stream_));

  // 3. cuBLAS DGEMV: dot[i] = centroids^T @ embedding
  double alpha = 1.0;
  double beta = 0.0;
  CUBLAS_CHECK(cublasDgemv(cublas_handle_, CUBLAS_OP_T,  // transpose centroids
                           dim_, n_clusters_,             // m, n
                           &alpha, d_centroids_, dim_,    // A, lda
                           d_embedding_, 1,               // x, incx
                           &beta, d_dots_, 1));           // y, incy

  // 4. Fused kernel: compute distances and find argmin in single kernel
  constexpr int kBlockSize = 256;  // Good for small n_clusters (20-50)
  fused_distance_argmin_kernel<double><<<1, kBlockSize, 0, stream_>>>(
      d_centroid_norms_, embed_norm, d_dots_, n_clusters_,
      d_best_idx_, d_best_dist_);

  // Check for kernel launch errors
  CUDA_CHECK(cudaGetLastError());

  // 5. Copy results back to host
  int host_best_idx;
  double host_best_dist;
  CUDA_CHECK(cudaMemcpyAsync(&host_best_idx, d_best_idx_, sizeof(int),
                            cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaMemcpyAsync(&host_best_dist, d_best_dist_, sizeof(double),
                            cudaMemcpyDeviceToHost, stream_));

  // 6. Synchronize to ensure copies complete
  CUDA_CHECK(cudaStreamSynchronize(stream_));

  return {host_best_idx, host_best_dist};
}

#endif  // NORDLYS_HAS_CUDA
