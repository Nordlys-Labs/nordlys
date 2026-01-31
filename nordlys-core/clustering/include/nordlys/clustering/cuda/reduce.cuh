#pragma once
#ifdef NORDLYS_HAS_CUDA

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <nordlys/clustering/cuda/common.cuh>

namespace nordlys::clustering::cuda {
namespace cg = cooperative_groups;

template <std::floating_point T>
__device__ __forceinline__ auto warp_reduce_sum(T val) noexcept -> T {
  auto warp = cg::tiled_partition<32>(cg::this_thread_block());
  return cg::reduce(warp, val, cg::plus<T>());
}

template <std::floating_point T>
__device__ __forceinline__ void warp_reduce_min_idx(T& val, int& idx) noexcept {
  auto warp = cg::tiled_partition<32>(cg::this_thread_block());
  
  for (int offset = 16; offset > 0; offset >>= 1) {
    T other_val = warp.shfl_down(val, offset);
    int other_idx = warp.shfl_down(idx, offset);
    if (other_val < val) {
      val = other_val;
      idx = other_idx;
    }
  }
}

template <std::floating_point T>
__device__ __forceinline__ auto block_reduce_sum_broadcast(T val, T* s_partial) noexcept -> T {
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

template <std::floating_point T>
__device__ __forceinline__ auto block_reduce_argmin(
    T& local_min, int& local_idx, T* s_min_dist,
    int* s_min_idx) noexcept -> bool {
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
    local_min = (lane < num_warps) ? s_min_dist[lane] : type_traits<T>.max_value;
    local_idx = (lane < num_warps) ? s_min_idx[lane] : -1;

    warp_reduce_min_idx(local_min, local_idx);

    return (lane == 0);
  }
  return false;
}

template <std::floating_point T>
__device__ __forceinline__ auto compute_partial_squared_norm(
    const T* __restrict__ vec, int dim) noexcept -> T {
  const int tid = threadIdx.x;
  T sum = T{0};

  const int vec_elements = dim / 4;
  
  if (vec_elements > 0) {
    const float4* vec4 = reinterpret_cast<const float4*>(vec);
    
    for (int i = tid; i < vec_elements; i += blockDim.x) {
      float4 v = __ldg(&vec4[i]);
      sum += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
  }

  for (int i = vec_elements * 4 + tid; i < dim; i += blockDim.x) {
    const T val = __ldg(&vec[i]);
    sum += val * val;
  }

  return sum;
}

template <std::floating_point T>
__device__ __forceinline__ void compute_partial_l2_argmin(
    const T* __restrict__ centroid_norms,
    const T* __restrict__ dots,
    int n_clusters, T q_norm,
    T& local_min, int& local_idx) noexcept {
  const int tid = threadIdx.x;
  constexpr T two = T{2};

  local_min = type_traits<T>.max_value;
  local_idx = -1;

  for (int i = tid; i < n_clusters; i += blockDim.x) {
    const T dist = __ldg(&centroid_norms[i]) + q_norm - two * __ldg(&dots[i]);
    if (dist < local_min) {
      local_min = dist;
      local_idx = i;
    }
  }
}

} // namespace nordlys::clustering::cuda

#endif
