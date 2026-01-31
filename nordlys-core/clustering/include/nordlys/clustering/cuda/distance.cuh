#pragma once
#ifdef NORDLYS_HAS_CUDA

#include <cmath>
#include <concepts>

#include <nordlys/clustering/cuda/reduce.cuh>

namespace nordlys::clustering::cuda {

// =============================================================================
// Types
// =============================================================================

template <std::floating_point T>
struct ClusterResult {
  int idx;
  T dist;
};

// =============================================================================
// Shared Memory Size Helpers
// =============================================================================

inline constexpr int kSharedMemPadding = 33;  // 32 + 1 padding for bank conflicts

template <std::floating_point T>
constexpr auto reduction_shared_mem_size() noexcept -> size_t {
  return kSharedMemPadding * sizeof(T) +      // s_partial
         kSharedMemPadding * sizeof(T) +      // s_min_dist
         kSharedMemPadding * sizeof(int);     // s_min_idx
}

template <std::floating_point T>
constexpr auto single_query_shared_mem_size(int n_clusters) noexcept -> size_t {
  int padded = ((n_clusters + 31) / 32) * 32;
  return padded * sizeof(T) + reduction_shared_mem_size<T>();
}

template <std::floating_point T>
constexpr auto batch_shared_mem_size() noexcept -> size_t {
  return reduction_shared_mem_size<T>();
}

// =============================================================================
// Single Query Kernel (GPU Buffer Path)
// =============================================================================

template <std::floating_point T>
__global__ void __launch_bounds__(256, 2)
find_nearest_centroid(
    const T* __restrict__ query,
    const T* __restrict__ centroids,
    const T* __restrict__ centroid_norms,
    int* __restrict__ out_idx,
    T* __restrict__ out_dist,
    int n_clusters,
    int dim
) {
  extern __shared__ char smem[];
  
  int padded = ((n_clusters + 31) / 32) * 32;
  T* s_dist = reinterpret_cast<T*>(smem);
  T* s_partial = s_dist + padded;
  T* s_min_dist = s_partial + kSharedMemPadding;
  int* s_min_idx = reinterpret_cast<int*>(s_min_dist + kSharedMemPadding);
  
  T q_norm = block_reduce_sum_broadcast(compute_partial_squared_norm(query, dim), s_partial);
  
  for (int c = threadIdx.x; c < n_clusters; c += blockDim.x) {
    const T* cent = centroids + c * dim;
    T dot = T{0};
    
    int vec_n = dim / 4;
    if (vec_n > 0) {
      const float4* q4 = reinterpret_cast<const float4*>(query);
      const float4* c4 = reinterpret_cast<const float4*>(cent);
      for (int i = 0; i < vec_n; ++i) {
        float4 qv = __ldg(&q4[i]);
        float4 cv = __ldg(&c4[i]);
        dot += qv.x*cv.x + qv.y*cv.y + qv.z*cv.z + qv.w*cv.w;
      }
    }
    for (int i = vec_n * 4; i < dim; ++i) {
      dot += __ldg(&query[i]) * __ldg(&cent[i]);
    }
    
    s_dist[c] = __ldg(&centroid_norms[c]) + q_norm - T{2} * dot;
  }
  __syncthreads();
  
  T best_val = type_traits<T>.max_value;
  int best_idx = -1;
  for (int i = threadIdx.x; i < n_clusters; i += blockDim.x) {
    if (s_dist[i] < best_val) {
      best_val = s_dist[i];
      best_idx = i;
    }
  }
  
  if (block_reduce_argmin(best_val, best_idx, s_min_dist, s_min_idx)) {
    *out_idx = best_idx;
    *out_dist = sqrt(best_val < T{0} ? T{0} : best_val);
  }
}

// =============================================================================
// Single Query Kernel (CPU Buffer Path - uses precomputed dots from cuBLAS)
// =============================================================================

template <std::floating_point T>
__global__ void find_nearest_centroid_with_dots(
    const T* __restrict__ query,
    const T* __restrict__ centroid_norms,
    const T* __restrict__ dots,
    int n_clusters,
    int dim,
    int* __restrict__ out_idx,
    T* __restrict__ out_dist
) {
  extern __shared__ char smem[];
  T* s_partial = reinterpret_cast<T*>(smem);
  T* s_min_dist = s_partial + kSharedMemPadding;
  int* s_min_idx = reinterpret_cast<int*>(s_min_dist + kSharedMemPadding);

  T q_norm = block_reduce_sum_broadcast(compute_partial_squared_norm(query, dim), s_partial);

  T best_val;
  int best_idx;
  compute_partial_l2_argmin(centroid_norms, dots, n_clusters, q_norm, best_val, best_idx);

  if (block_reduce_argmin(best_val, best_idx, s_min_dist, s_min_idx)) {
    *out_idx = best_idx;
    *out_dist = sqrt(best_val < T{0} ? T{0} : best_val);
  }
}

// =============================================================================
// Batch Query Kernel
// =============================================================================

template <std::floating_point T>
__global__ void find_nearest_centroid_batch(
    const T* __restrict__ queries,
    const T* __restrict__ centroid_norms,
    const T* __restrict__ dots,
    int n_queries,
    int n_clusters,
    int dim,
    ClusterResult<T>* __restrict__ results
) {
  int qid = blockIdx.x;
  if (qid >= n_queries) return;

  extern __shared__ char smem[];
  T* s_partial = reinterpret_cast<T*>(smem);
  T* s_min_dist = s_partial + kSharedMemPadding;
  int* s_min_idx = reinterpret_cast<int*>(s_min_dist + kSharedMemPadding);

  const T* query = queries + qid * dim;
  const T* query_dots = dots + qid * n_clusters;

  T q_norm = block_reduce_sum_broadcast(compute_partial_squared_norm(query, dim), s_partial);

  T best_val;
  int best_idx;
  compute_partial_l2_argmin(centroid_norms, query_dots, n_clusters, q_norm, best_val, best_idx);

  if (block_reduce_argmin(best_val, best_idx, s_min_dist, s_min_idx)) {
    results[qid] = {best_idx, sqrt(best_val < T{0} ? T{0} : best_val)};
  }
}

} // namespace nordlys::clustering::cuda

#endif
