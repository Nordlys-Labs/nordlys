#include <climits>
#include <cmath>
#include <cstring>
#include <limits>
#include <nordlys/clustering/cluster_cpu.hpp>

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <usearch/index_dense.hpp>

namespace {

  struct MinDistanceResult {
    float dist_sq;
    int idx;
  };

#ifdef _OPENMP
#  ifndef _MSC_VER
#    pragma omp declare reduction(custom_min_float:MinDistanceResult : omp_out                 \
                                      = (omp_in.dist_sq < omp_out.dist_sq) ? omp_in : omp_out) \
        initializer(omp_priv = {std::numeric_limits<float>::max(), -1})
#  endif
#endif

}  // namespace

void CpuClusterBackend::load_centroids(const float* data, size_t n_clusters, size_t dim) {
  if (n_clusters == 0 || dim == 0) [[unlikely]] {
    throw std::invalid_argument("load_centroids: n_clusters and dim must be non-zero");
  }

  if (n_clusters > static_cast<size_t>(INT_MAX)) [[unlikely]] {
    throw std::invalid_argument("load_centroids: n_clusters exceeds INT_MAX");
  }

  if (dim > static_cast<size_t>(INT_MAX)) [[unlikely]] {
    throw std::invalid_argument("load_centroids: dim exceeds INT_MAX");
  }

  if (data == nullptr) [[unlikely]] {
    throw std::invalid_argument("load_centroids: data pointer is null");
  }

  if (n_clusters > SIZE_MAX / dim) [[unlikely]] {
    throw std::invalid_argument("load_centroids: n_clusters * dim would overflow");
  }

  size_t total_size = n_clusters * dim;
  if (total_size > SIZE_MAX / sizeof(float)) [[unlikely]] {
    throw std::invalid_argument("load_centroids: allocation size would overflow");
  }

  n_clusters_ = static_cast<int>(n_clusters);
  dim_ = static_cast<int>(dim);

  using namespace unum::usearch;
  metric_ = metric_punned_t(dim, metric_kind_t::l2sq_k, scalar_kind_t::f32_k);

  centroids_.resize(total_size);
  std::memcpy(centroids_.data(), data, centroids_.size() * sizeof(float));
}

std::pair<int, float> CpuClusterBackend::assign(EmbeddingView view) {
  if (n_clusters_ == 0) return {-1, 0.0f};
  if (view.dim != static_cast<size_t>(dim_)) [[unlikely]] {
    throw std::invalid_argument("assign: dimension mismatch");
  }

  std::visit(overloaded{[](CpuDevice) {},
                        [](CudaDevice) -> void {
                          throw std::invalid_argument("assign: GPU tensor passed to CPU backend");
                        }},
             view.device);

  const auto* emb_bytes = reinterpret_cast<const unum::usearch::byte_t*>(view.data);

  int best_idx = -1;
  float best_dist_sq = std::numeric_limits<float>::max();

#ifdef _OPENMP
  if (n_clusters_ > 100) {
#  ifdef _MSC_VER
#    pragma omp parallel for
    for (int i = 0; i < n_clusters_; ++i) {
      const auto* centroid_bytes
          = reinterpret_cast<const unum::usearch::byte_t*>(centroids_.data() + i * dim_);
      auto dist_sq = static_cast<float>(metric_(emb_bytes, centroid_bytes));

#    pragma omp critical
      {
        if (dist_sq < best_dist_sq) {
          best_dist_sq = dist_sq;
          best_idx = i;
        }
      }
    }
#  else
    MinDistanceResult result{std::numeric_limits<float>::max(), -1};

#    pragma omp parallel for reduction(custom_min_float : result)
    for (int i = 0; i < n_clusters_; ++i) {
      const auto* centroid_bytes
          = reinterpret_cast<const unum::usearch::byte_t*>(centroids_.data() + i * dim_);
      auto dist_sq = static_cast<float>(metric_(emb_bytes, centroid_bytes));

      if (dist_sq < result.dist_sq) {
        result.dist_sq = dist_sq;
        result.idx = i;
      }
    }

    best_dist_sq = result.dist_sq;
    best_idx = result.idx;
#  endif
  } else {
    for (int i = 0; i < n_clusters_; ++i) {
      const auto* centroid_bytes
          = reinterpret_cast<const unum::usearch::byte_t*>(centroids_.data() + i * dim_);
      auto dist_sq = static_cast<float>(metric_(emb_bytes, centroid_bytes));

      if (dist_sq < best_dist_sq) {
        best_dist_sq = dist_sq;
        best_idx = i;
      }
    }
  }
#else
  for (int i = 0; i < n_clusters_; ++i) {
    const auto* centroid_bytes
        = reinterpret_cast<const unum::usearch::byte_t*>(centroids_.data() + i * dim_);
    auto dist_sq = static_cast<float>(metric_(emb_bytes, centroid_bytes));

    if (dist_sq < best_dist_sq) {
      best_dist_sq = dist_sq;
      best_idx = i;
    }
  }
#endif

  return {best_idx, std::sqrt(best_dist_sq)};
}

std::vector<std::pair<int, float>> CpuClusterBackend::assign_batch(EmbeddingBatchView view) {
  if (n_clusters_ > 0 && view.dim != static_cast<size_t>(dim_)) [[unlikely]] {
    throw std::invalid_argument("assign_batch: dimension mismatch");
  }

  std::visit(
      overloaded{[](CpuDevice) {},
                 [](CudaDevice) -> void {
                   throw std::invalid_argument("assign_batch: GPU tensor passed to CPU backend");
                 }},
      view.device);

  std::vector<std::pair<int, float>> results(view.count);

#ifdef _OPENMP
#  pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < static_cast<int>(view.count); ++i) {
    size_t idx = static_cast<size_t>(i);
    EmbeddingView single_view{view.data + idx * view.dim, view.dim, view.device};
    results[idx] = assign(single_view);
  }

  return results;
}
