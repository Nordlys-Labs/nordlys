#ifdef ADAPTIVE_HAS_CUDA

#include "cluster_cuda.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include <cmath>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template<typename Scalar>
std::vector<Scalar> to_col_major(const Scalar* data, size_t n_clusters, size_t dim) {
  std::vector<Scalar> result(n_clusters * dim);
  const size_t total = n_clusters * dim;
  std::ranges::for_each(std::views::iota(size_t{0}, total), [&](size_t idx) {
    const size_t d = idx / dim;
    const size_t k = idx % dim;
    result[idx] = data[k * dim + d];
  });
  return result;
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

// Epilogue kernel: Compute dist²[i] = ||c_i||² + ||e||² - 2*dot[i]
template<typename Scalar>
__global__ void epilogue_kernel(const Scalar* c_norms, Scalar e_norm, const Scalar* dots,
                                 Scalar* dists, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dists[i] = c_norms[i] + e_norm - Scalar(2.0) * dots[i];
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
  if (d_distances_) {
    cudaFree(d_distances_);
    d_distances_ = nullptr;
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
  CUDA_CHECK(cudaMalloc(&d_distances_, n_clusters_sz * sizeof(float)));

  // Convert row-major [K][D] to column-major [D][K] for cuBLAS
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

  // 4. Epilogue kernel: dist²[i] = ||c_i||² + ||e||² - 2*dot[i]
  constexpr int kBlockSize = 256;
  const unsigned int grid_size = static_cast<unsigned int>(
      (n_clusters_ + kBlockSize - 1) / kBlockSize);
   epilogue_kernel<float><<<grid_size, kBlockSize, 0, stream_>>>(
       d_centroid_norms_, embed_norm, d_dots_, d_distances_, n_clusters_);

   // Check for kernel launch errors
   CUDA_CHECK(cudaGetLastError());

   // 5. Argmin using Thrust on the same stream
   thrust::device_ptr<float> ptr(d_distances_);
   auto min_it = thrust::min_element(thrust::cuda::par.on(stream_), ptr, ptr + n_clusters_);

   // Explicitly copy the minimum value from device to host
   float host_min_value;
   thrust::copy(thrust::cuda::par.on(stream_), min_it, min_it + 1, &host_min_value);

   // 6. Synchronize to ensure Thrust completes before reading result
   CUDA_CHECK(cudaStreamSynchronize(stream_));
  int best = static_cast<int>(min_it - ptr);
  float best_dist = std::sqrt(std::max(host_min_value, 0.0f));

  return {best, best_dist};
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
  if (d_distances_) {
    cudaFree(d_distances_);
    d_distances_ = nullptr;
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
  CUDA_CHECK(cudaMalloc(&d_distances_, n_clusters_sz * sizeof(double)));

  // Convert row-major [K][D] to column-major [D][K] for cuBLAS
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

  // 4. Epilogue kernel: dist²[i] = ||c_i||² + ||e||² - 2*dot[i]
  constexpr int kBlockSize = 256;
  const unsigned int grid_size = static_cast<unsigned int>(
      (n_clusters_ + kBlockSize - 1) / kBlockSize);
   epilogue_kernel<double><<<grid_size, kBlockSize, 0, stream_>>>(
       d_centroid_norms_, embed_norm, d_dots_, d_distances_, n_clusters_);

   // Check for kernel launch errors
   CUDA_CHECK(cudaGetLastError());

   // 5. Argmin using Thrust on the same stream
   thrust::device_ptr<double> ptr(d_distances_);
   auto min_it = thrust::min_element(thrust::cuda::par.on(stream_), ptr, ptr + n_clusters_);

   // Explicitly copy the minimum value from device to host
   double host_min_value;
   thrust::copy(thrust::cuda::par.on(stream_), min_it, min_it + 1, &host_min_value);

   // 6. Synchronize to ensure Thrust completes before reading result
   CUDA_CHECK(cudaStreamSynchronize(stream_));
  int best = static_cast<int>(min_it - ptr);
  double best_dist = std::sqrt(std::max(host_min_value, 0.0));

  return {best, best_dist};
}

#endif  // ADAPTIVE_HAS_CUDA
