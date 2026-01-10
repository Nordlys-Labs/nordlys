#pragma once

#ifdef NORDLYS_HAS_CUDA

#include <cuda_runtime.h>
#include <limits>

namespace nordlys::cuda {

// Warp-level sum reduction
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// Compute squared L2 norm: ||v||²
template <typename T>
__global__ void compute_squared_norm(const T* __restrict__ vec, int dim, T* __restrict__ out) {
  __shared__ T s_partial[32];
  
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  
  T sum = T{0};
  
  // Vectorized: process 4 elements per iteration
  const int vec_dim = (dim / 4) * 4;
  for (int i = tid * 4; i < vec_dim; i += blockDim.x * 4) {
    if (i + 3 < dim) {
      const T v0 = vec[i];
      const T v1 = vec[i + 1];
      const T v2 = vec[i + 2];
      const T v3 = vec[i + 3];
      sum += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
    }
  }
  
  // Handle remainder
  for (int i = vec_dim + tid; i < dim; i += blockDim.x) {
    const T val = vec[i];
    sum += val * val;
  }
  
  sum = warp_reduce_sum(sum);
  
  if (lane == 0) s_partial[warp_id] = sum;
  __syncthreads();
  
  if (warp_id == 0) {
    const int num_warps = (blockDim.x + 31) >> 5;
    sum = (lane < num_warps) ? s_partial[lane] : T{0};
    sum = warp_reduce_sum(sum);
    if (lane == 0) *out = sum;
  }
}

// Fused L2 distance + argmin using precomputed norms
// Distance: ||a-b||² = ||a||² + ||b||² - 2(a·b)
template <typename T>
__global__ void fused_l2_argmin(
    const T* __restrict__ centroid_norms,  // [n]
    const T* __restrict__ query_norm,      // [1]
    const T* __restrict__ dots,            // [n] dot products
    int n,
    int* __restrict__ best_idx,
    T* __restrict__ best_dist) {

  __shared__ T s_min_dist[32];
  __shared__ int s_min_idx[32];

  const T q_norm = *query_norm;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;

  T local_min = cuda::std::numeric_limits<T>::infinity();
  int local_idx = -1;

  // ILP: process 4 distances per iteration
  constexpr T two = T{2};
  const int vec_end = (n / 4) * 4;
  
  for (int i = tid * 4; i < vec_end; i += blockDim.x * 4) {
    const T d0 = centroid_norms[i] + q_norm - two * dots[i];
    const T d1 = centroid_norms[i + 1] + q_norm - two * dots[i + 1];
    const T d2 = centroid_norms[i + 2] + q_norm - two * dots[i + 2];
    const T d3 = centroid_norms[i + 3] + q_norm - two * dots[i + 3];
    
    if (d0 < local_min) { local_min = d0; local_idx = i; }
    if (d1 < local_min) { local_min = d1; local_idx = i + 1; }
    if (d2 < local_min) { local_min = d2; local_idx = i + 2; }
    if (d3 < local_min) { local_min = d3; local_idx = i + 3; }
  }
  
  // Handle remainder
  for (int i = vec_end + tid; i < n; i += blockDim.x) {
    const T dist = centroid_norms[i] + q_norm - two * dots[i];
    if (dist < local_min) {
      local_min = dist;
      local_idx = i;
    }
  }

  // Warp-level reduction
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const T other_dist = __shfl_down_sync(0xffffffff, local_min, offset);
    const int other_idx = __shfl_down_sync(0xffffffff, local_idx, offset);
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

  // Final reduction across warps
  if (warp_id == 0) {
    const int num_warps = (blockDim.x + 31) >> 5;
    local_min = (lane < num_warps) ? s_min_dist[lane] : cuda::std::numeric_limits<T>::infinity();
    local_idx = (lane < num_warps) ? s_min_idx[lane] : -1;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      const T other_dist = __shfl_down_sync(0xffffffff, local_min, offset);
      const int other_idx = __shfl_down_sync(0xffffffff, local_idx, offset);
      if (other_dist < local_min) {
        local_min = other_dist;
        local_idx = other_idx;
      }
    }

    if (lane == 0) {
      *best_idx = local_idx;
      *best_dist = sqrt(local_min < T{0} ? T{0} : local_min);
    }
  }
}

}  // namespace nordlys::cuda

#endif  // NORDLYS_HAS_CUDA
