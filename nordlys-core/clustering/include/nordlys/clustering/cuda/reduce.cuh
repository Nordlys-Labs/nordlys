#pragma once
#ifdef NORDLYS_HAS_CUDA

#  include <cuda_runtime.h>

#  include <nordlys/clustering/cuda/common.cuh>

// =============================================================================
// Warp-level primitives
// =============================================================================

template <typename T> __device__ __forceinline__ T warp_reduce_sum(T val) {
#  pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

template <typename T> __device__ __forceinline__ void warp_reduce_min_idx(T& val, int& idx) {
#  pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    T other_val = __shfl_down_sync(0xffffffff, val, offset);
    int other_idx = __shfl_down_sync(0xffffffff, idx, offset);
    if (other_val < val) {
      val = other_val;
      idx = other_idx;
    }
  }
}

// =============================================================================
// Block-level primitives
// =============================================================================

// Block-level sum reduction using shared memory
// Returns the sum in thread 0, broadcasts to all threads via shared memory
// s_partial must have at least 32 elements
template <typename T> __device__ __forceinline__ T block_reduce_sum_broadcast(T val, T* s_partial) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;

  val = warp_reduce_sum(val);

  if (lane == 0) s_partial[warp_id] = val;
  __syncthreads();

  if (warp_id == 0) {
    const int num_warps = (blockDim.x + 31) >> 5;
    val = (lane < num_warps) ? s_partial[lane] : T{0};
    val = warp_reduce_sum(val);
    if (lane == 0) s_partial[0] = val;
  }
  __syncthreads();

  return s_partial[0];
}

// Block-level argmin reduction
// s_min_dist and s_min_idx must have at least 32 elements each
// Result is written by thread 0 only; returns true for thread 0
template <typename T>
__device__ __forceinline__ bool block_reduce_argmin(T& local_min, int& local_idx, T* s_min_dist,
                                                    int* s_min_idx) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;

  warp_reduce_min_idx(local_min, local_idx);

  if (lane == 0) {
    s_min_dist[warp_id] = local_min;
    s_min_idx[warp_id] = local_idx;
  }
  __syncthreads();

  if (warp_id == 0) {
    const int num_warps = (blockDim.x + 31) >> 5;
    local_min = (lane < num_warps) ? s_min_dist[lane] : CudaTypeTraits<T>::max_value;
    local_idx = (lane < num_warps) ? s_min_idx[lane] : -1;

    warp_reduce_min_idx(local_min, local_idx);

    return (lane == 0);
  }
  return false;
}

// =============================================================================
// Vectorized computation primitives
// =============================================================================

// Compute partial squared norm of a vector using vectorized loads
// Each thread computes its portion; caller must reduce across block
template <typename T>
__device__ __forceinline__ T compute_partial_squared_norm(const T* __restrict__ vec, int dim) {
  const int tid = threadIdx.x;
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

  return sum;
}

// Find argmin of L2 distances: dist[i] = centroid_norms[i] + q_norm - 2*dots[i]
// Each thread computes its portion; caller must reduce across block
template <typename T>
__device__ __forceinline__ void compute_partial_l2_argmin(const T* __restrict__ centroid_norms,
                                                          const T* __restrict__ dots,
                                                          int n_clusters, T q_norm, T& local_min,
                                                          int& local_idx) {
  const int tid = threadIdx.x;
  constexpr T two = T{2};

  local_min = CudaTypeTraits<T>::max_value;
  local_idx = -1;

  for (int i = tid; i < n_clusters; i += blockDim.x) {
    const T dist = centroid_norms[i] + q_norm - two * dots[i];
    if (dist < local_min) {
      local_min = dist;
      local_idx = i;
    }
  }
}

#endif
