#pragma once
#ifdef NORDLYS_HAS_CUDA

#  include <cmath>
#  include <nordlys/clustering/cuda/common.cuh>
#  include <nordlys/clustering/cuda/reduce.cuh>
#  include <vector>

// =============================================================================
// Host utilities
// =============================================================================

template <typename T>
inline std::vector<T> to_col_major(const T* row_major, size_t rows, size_t cols) {
  std::vector<T> col_major(rows * cols);
  for (size_t r = 0; r < rows; ++r) {
    for (size_t c = 0; c < cols; ++c) {
      col_major[c * rows + r] = row_major[r * cols + c];
    }
  }
  return col_major;
}

// =============================================================================
// Shared memory size calculation
// =============================================================================

// Returns required shared memory size for fused L2 argmin kernels
template <typename T> inline size_t fused_l2_argmin_shared_mem_size() {
  // s_partial[32] + s_min_dist[32] + s_min_idx[32]
  return 32 * sizeof(T) + 32 * sizeof(T) + 32 * sizeof(int);
}

// =============================================================================
// Kernels
// =============================================================================

// Single-query fused kernel: computes query norm + L2 argmin in one pass
// Launch with: <<<1, block_size, fused_l2_argmin_shared_mem_size<T>(), stream>>>
template <typename T>
__global__ void fused_l2_argmin_with_norm(const T* __restrict__ query,
                                          const T* __restrict__ centroid_norms,
                                          const T* __restrict__ dots, int n_clusters, int dim,
                                          int* __restrict__ best_idx, T* __restrict__ best_dist) {
  extern __shared__ char shared_mem[];
  T* s_partial = reinterpret_cast<T*>(shared_mem);
  T* s_min_dist = s_partial + 32;
  int* s_min_idx = reinterpret_cast<int*>(s_min_dist + 32);

  // Phase 1: Compute query norm
  T partial_norm = compute_partial_squared_norm(query, dim);
  T q_norm = block_reduce_sum_broadcast(partial_norm, s_partial);

  // Phase 2: Find argmin L2 distance
  T local_min;
  int local_idx;
  compute_partial_l2_argmin(centroid_norms, dots, n_clusters, q_norm, local_min, local_idx);

  if (block_reduce_argmin(local_min, local_idx, s_min_dist, s_min_idx)) {
    *best_idx = local_idx;
    *best_dist = sqrt(local_min < T{0} ? T{0} : local_min);
  }
}

// Batch fused kernel: one block per query, computes norm + argmin
// Launch with: <<<n_queries, block_size, fused_l2_argmin_shared_mem_size<T>(), stream>>>
template <typename T> __global__ void batch_l2_argmin_fused(
    const T* __restrict__ queries, const T* __restrict__ centroid_norms, const T* __restrict__ dots,
    int n_queries, int n_clusters, int dim, int* __restrict__ best_idx, T* __restrict__ best_dist) {
  int query_id = blockIdx.x;
  if (query_id >= n_queries) return;

  extern __shared__ char shared_mem[];
  T* s_partial = reinterpret_cast<T*>(shared_mem);
  T* s_min_dist = s_partial + 32;
  int* s_min_idx = reinterpret_cast<int*>(s_min_dist + 32);

  const T* query = queries + query_id * dim;
  const T* query_dots = dots + query_id * n_clusters;

  // Phase 1: Compute query norm
  T partial_norm = compute_partial_squared_norm(query, dim);
  T q_norm = block_reduce_sum_broadcast(partial_norm, s_partial);

  // Phase 2: Find argmin L2 distance
  T local_min;
  int local_idx;
  compute_partial_l2_argmin(centroid_norms, query_dots, n_clusters, q_norm, local_min, local_idx);

  if (block_reduce_argmin(local_min, local_idx, s_min_dist, s_min_idx)) {
    best_idx[query_id] = local_idx;
    best_dist[query_id] = sqrt(local_min < T{0} ? T{0} : local_min);
  }
}

#endif
