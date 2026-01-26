#include <cmath>
#include <cstring>
#include <limits>
#include <nordlys/clustering/cluster.hpp>
#include <type_traits>

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <usearch/index.hpp>
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

// =============================================================================
// CPU Backend Implementation
// =============================================================================

class CpuClusterBackend : public IClusterBackend {
public:
  void load_centroids(const float* data, size_t n_clusters, size_t dim) override {
    if (n_clusters <= 0 || dim <= 0) [[unlikely]] {
      throw std::invalid_argument("n_clusters and dim must be positive");
    }

    auto nc = static_cast<size_t>(n_clusters);
    auto d = static_cast<size_t>(dim);

    if (nc > SIZE_MAX / d) [[unlikely]] {
      throw std::invalid_argument("n_clusters * dim would overflow");
    }

    size_t total_size = nc * d;
    if (total_size > SIZE_MAX / sizeof(float)) [[unlikely]] {
      throw std::invalid_argument("allocation size would overflow");
    }

    n_clusters_ = static_cast<int>(n_clusters);
    dim_ = static_cast<int>(dim);

    using namespace unum::usearch;
    metric_ = metric_punned_t(d, metric_kind_t::l2sq_k, scalar_kind_t::f32_k);

    centroids_.resize(total_size);
    std::memcpy(centroids_.data(), data, centroids_.size() * sizeof(float));
  }

  [[nodiscard]] std::pair<int, float> assign(EmbeddingView view) override {
    if (n_clusters_ == 0) return {-1, 0.0f};
    if (view.dim != static_cast<size_t>(dim_)) [[unlikely]] {
      throw std::invalid_argument("dimension mismatch in assign");
    }

    std::visit(overloaded{[](CpuDevice) {},
                          [](CudaDevice) -> void {
                            throw std::invalid_argument(
                                "GPU tensor passed to CPU backend. "
                                "Create Nordlys with device=CudaDevice{} to use GPU embeddings.");
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

  [[nodiscard]] std::vector<std::pair<int, float>> assign_batch(EmbeddingBatchView view) override {
    if (n_clusters_ > 0 && view.dim != static_cast<size_t>(dim_)) [[unlikely]] {
      throw std::invalid_argument("dimension mismatch in assign_batch");
    }

    std::visit(overloaded{[](CpuDevice) {},
                          [](CudaDevice) -> void {
                            throw std::invalid_argument(
                                "GPU tensor passed to CPU backend. "
                                "Create Nordlys with device=CudaDevice{} to use GPU embeddings.");
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

  [[nodiscard]] size_t n_clusters() const noexcept override {
    return static_cast<size_t>(n_clusters_);
  }
  [[nodiscard]] size_t dim() const noexcept override { return static_cast<size_t>(dim_); }

private:
  std::vector<float> centroids_;
  unum::usearch::metric_punned_t metric_;
  int n_clusters_ = 0;
  int dim_ = 0;
};

// =============================================================================
// Factory Functions
// =============================================================================

bool cuda_available() noexcept {
#ifdef NORDLYS_HAS_CUDA
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  return err == cudaSuccess && device_count > 0;
#else
  return false;
#endif
}

std::unique_ptr<IClusterBackend> create_backend(Device device) {
  return std::visit(overloaded{[](CpuDevice) -> std::unique_ptr<IClusterBackend> {
                                 return std::make_unique<CpuClusterBackend>();
                               },
                               [](CudaDevice) -> std::unique_ptr<IClusterBackend> {
#ifdef NORDLYS_HAS_CUDA
                                 if (cuda_available()) {
                                   return std::make_unique<CudaClusterBackend>();
                                 }
                                 throw std::runtime_error("CUDA not available");
#else
                                 throw std::runtime_error("CUDA backend not compiled");
#endif
                               }},
                    device);
}
