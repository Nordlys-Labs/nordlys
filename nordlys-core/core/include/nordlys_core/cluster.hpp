#pragma once

#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef _OPENMP
#  include <omp.h>
#endif

#include "device.hpp"
#include "embedding_view.hpp"
#include "matrix.hpp"

template <typename Scalar> class IClusterBackend {
public:
  virtual ~IClusterBackend() = default;

  IClusterBackend(const IClusterBackend&) = delete;
  IClusterBackend& operator=(const IClusterBackend&) = delete;
  IClusterBackend(IClusterBackend&&) = delete;
  IClusterBackend& operator=(IClusterBackend&&) = delete;

  virtual void load_centroids(const Scalar* data, size_t n_clusters, size_t dim) = 0;

  [[nodiscard]] virtual std::pair<int, Scalar> assign(EmbeddingView<Scalar> view) = 0;

  [[nodiscard]] virtual std::vector<std::pair<int, Scalar>> assign_batch(
      EmbeddingBatchView<Scalar> view) = 0;

  [[nodiscard]] virtual size_t n_clusters() const = 0;
  [[nodiscard]] virtual size_t dim() const = 0;

protected:
  IClusterBackend() = default;
};

// =============================================================================
// CPU Backend (excluded from CUDA compilation - usearch not compatible with nvcc)
// =============================================================================

#ifndef __CUDACC__

#  include <usearch/index.hpp>
#  include <usearch/index_dense.hpp>

// Helper struct for OpenMP custom reduction
template <typename Scalar> struct MinDistanceResult {
  Scalar dist_sq;
  int idx;
};

// Declare custom OpenMP reduction for MinDistanceResult
// Note: We declare explicit reductions for float and double to ensure compatibility
// Template-based reductions have limited compiler support
// MSVC's OpenMP 2.0 doesn't support custom reductions (OpenMP 4.0+ feature)
#  ifdef _OPENMP
#    ifndef _MSC_VER
// For float
#      pragma omp declare reduction(custom_min_float:MinDistanceResult<float>: \
        omp_out = (omp_in.dist_sq < omp_out.dist_sq) ? omp_in : omp_out) \
        initializer(omp_priv = {std::numeric_limits<float>::max(), -1})

// For double  
#      pragma omp declare reduction(custom_min_double:MinDistanceResult<double>: \
        omp_out = (omp_in.dist_sq < omp_out.dist_sq) ? omp_in : omp_out) \
        initializer(omp_priv = {std::numeric_limits<double>::max(), -1})
#    endif  // _MSC_VER
#  endif

template <typename Scalar> class CpuClusterBackend : public IClusterBackend<Scalar> {
public:
  void load_centroids(const Scalar* data, size_t n_clusters, size_t dim) override {
    if (n_clusters <= 0 || dim <= 0) [[unlikely]] {
      throw std::invalid_argument("n_clusters and dim must be positive");
    }

    auto nc = static_cast<size_t>(n_clusters);
    auto d = static_cast<size_t>(dim);

    if (nc > SIZE_MAX / d) [[unlikely]] {
      throw std::invalid_argument("n_clusters * dim would overflow");
    }

    size_t total_size = nc * d;
    if (total_size > SIZE_MAX / sizeof(Scalar)) [[unlikely]] {
      throw std::invalid_argument("allocation size would overflow");
    }

    n_clusters_ = static_cast<int>(n_clusters);
    dim_ = static_cast<int>(dim);

    using namespace unum::usearch;
    auto scalar_kind = std::is_same_v<Scalar, float> ? scalar_kind_t::f32_k : scalar_kind_t::f64_k;
    metric_ = metric_punned_t(d, metric_kind_t::l2sq_k, scalar_kind);

    centroids_.resize(total_size);
    std::memcpy(centroids_.data(), data, centroids_.size() * sizeof(Scalar));
  }

  [[nodiscard]] std::pair<int, Scalar> assign(EmbeddingView<Scalar> view) override {
    if (n_clusters_ == 0) return {-1, Scalar{0}};
    if (view.dim != static_cast<size_t>(dim_)) [[unlikely]] {
      throw std::invalid_argument("dimension mismatch in assign");
    }

    // Throw if GPU memory passed to CPU backend
    std::visit(overloaded{
        [](CpuDevice) {},
        [](CudaDevice) -> void {
            throw std::invalid_argument("GPU tensor passed to CPU backend. "
                                      "Create Nordlys with device=CudaDevice{} to use GPU embeddings.");
        }
    }, view.device);

    const auto* emb_bytes = reinterpret_cast<const unum::usearch::byte_t*>(view.data);

    int best_idx = -1;
    Scalar best_dist_sq = std::numeric_limits<Scalar>::max();

#  ifdef _OPENMP
    if (n_clusters_ > 100) {
#    ifdef _MSC_VER
      // MSVC OpenMP 2.0 doesn't support custom reductions
      // Use critical section for thread-safe min finding
#      pragma omp parallel for
      for (int i = 0; i < n_clusters_; ++i) {
        const auto* centroid_bytes
            = reinterpret_cast<const unum::usearch::byte_t*>(centroids_.data() + i * dim_);
        auto dist_sq = static_cast<Scalar>(metric_(emb_bytes, centroid_bytes));

#        pragma omp critical
        {
          if (dist_sq < best_dist_sq) {
            best_dist_sq = dist_sq;
            best_idx = i;
          }
        }
      }
#    else  // GCC/Clang: Use custom reduction (OpenMP 4.0+)
      MinDistanceResult<Scalar> result{std::numeric_limits<Scalar>::max(), -1};

      if constexpr (std::is_same_v<Scalar, float>) {
#        pragma omp parallel for reduction(custom_min_float:result)
        for (int i = 0; i < n_clusters_; ++i) {
          const auto* centroid_bytes
              = reinterpret_cast<const unum::usearch::byte_t*>(centroids_.data() + i * dim_);
          auto dist_sq = static_cast<Scalar>(metric_(emb_bytes, centroid_bytes));

          if (dist_sq < result.dist_sq) {
            result.dist_sq = dist_sq;
            result.idx = i;
          }
        }
      } else if constexpr (std::is_same_v<Scalar, double>) {
#        pragma omp parallel for reduction(custom_min_double:result)
        for (int i = 0; i < n_clusters_; ++i) {
          const auto* centroid_bytes
              = reinterpret_cast<const unum::usearch::byte_t*>(centroids_.data() + i * dim_);
          auto dist_sq = static_cast<Scalar>(metric_(emb_bytes, centroid_bytes));

          if (dist_sq < result.dist_sq) {
            result.dist_sq = dist_sq;
            result.idx = i;
          }
        }
      } else {
        // Fallback for other types: use sequential or critical section
        for (int i = 0; i < n_clusters_; ++i) {
          const auto* centroid_bytes
              = reinterpret_cast<const unum::usearch::byte_t*>(centroids_.data() + i * dim_);
          auto dist_sq = static_cast<Scalar>(metric_(emb_bytes, centroid_bytes));

          if (dist_sq < result.dist_sq) {
            result.dist_sq = dist_sq;
            result.idx = i;
          }
        }
      }

      best_dist_sq = result.dist_sq;
      best_idx = result.idx;
#    endif  // _MSC_VER
    } else {
      for (int i = 0; i < n_clusters_; ++i) {
        const auto* centroid_bytes
            = reinterpret_cast<const unum::usearch::byte_t*>(centroids_.data() + i * dim_);
        auto dist_sq = static_cast<Scalar>(metric_(emb_bytes, centroid_bytes));

        if (dist_sq < best_dist_sq) {
          best_dist_sq = dist_sq;
          best_idx = i;
        }
      }
    }
#  else
    for (int i = 0; i < n_clusters_; ++i) {
      const auto* centroid_bytes
          = reinterpret_cast<const unum::usearch::byte_t*>(centroids_.data() + i * dim_);
      auto dist_sq = static_cast<Scalar>(metric_(emb_bytes, centroid_bytes));

      if (dist_sq < best_dist_sq) {
        best_dist_sq = dist_sq;
        best_idx = i;
      }
    }
#  endif

    return {best_idx, std::sqrt(best_dist_sq)};
  }

  [[nodiscard]] std::vector<std::pair<int, Scalar>> assign_batch(
      EmbeddingBatchView<Scalar> view) override {
    if (n_clusters_ > 0 && view.dim != static_cast<size_t>(dim_)) [[unlikely]] {
      throw std::invalid_argument("dimension mismatch in assign_batch");
    }

    // Throw if GPU memory passed to CPU backend
    std::visit(overloaded{
        [](CpuDevice) {},
        [](CudaDevice) -> void {
            throw std::invalid_argument("GPU tensor passed to CPU backend. "
                                      "Create Nordlys with device=CudaDevice{} to use GPU embeddings.");
        }
    }, view.device);

    std::vector<std::pair<int, Scalar>> results(view.count);

#  ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#  endif
    for (int i = 0; i < static_cast<int>(view.count); ++i) {
      size_t idx = static_cast<size_t>(i);
      EmbeddingView<Scalar> single_view{
          view.data + idx * view.dim,
          view.dim,
          view.device
      };
      results[idx] = assign(single_view);
    }

    return results;
  }

  [[nodiscard]] size_t n_clusters() const noexcept override { return static_cast<size_t>(n_clusters_); }
  [[nodiscard]] size_t dim() const noexcept override { return static_cast<size_t>(dim_); }

private:
  std::vector<Scalar> centroids_;
  unum::usearch::metric_punned_t metric_;
  int n_clusters_ = 0;
  int dim_ = 0;
};

#endif  // __CUDACC__

// =============================================================================
// CUDA Backend (only available when CUDA is enabled)
// =============================================================================

#ifdef NORDLYS_HAS_CUDA

#  include <cublas_v2.h>
#  include <cuda_runtime.h>

#  include <nordlys_core/cuda_memory.cuh>

template <typename Scalar> class CudaClusterBackend : public IClusterBackend<Scalar> {
public:
  CudaClusterBackend();
  ~CudaClusterBackend() override;

  CudaClusterBackend(const CudaClusterBackend&) = delete;
  CudaClusterBackend& operator=(const CudaClusterBackend&) = delete;
  CudaClusterBackend(CudaClusterBackend&&) = delete;
  CudaClusterBackend& operator=(CudaClusterBackend&&) = delete;

  void load_centroids(const Scalar* data, size_t n_clusters, size_t dim) override;

  [[nodiscard]] std::pair<int, Scalar> assign(EmbeddingView<Scalar> view) override;

  [[nodiscard]] std::vector<std::pair<int, Scalar>> assign_batch(
      EmbeddingBatchView<Scalar> view) override;

  [[nodiscard]] size_t n_clusters() const noexcept override { return static_cast<size_t>(n_clusters_); }
  [[nodiscard]] size_t dim() const noexcept override { return static_cast<size_t>(dim_); }

private:
  void free_memory();
  void capture_graph();
  void ensure_batch_capacity(int count);
  
  // Helper methods for device-aware batch processing
  std::vector<std::pair<int, Scalar>> assign_batch_from_host(EmbeddingBatchView<Scalar> view);
  std::vector<std::pair<int, Scalar>> assign_batch_from_device(EmbeddingBatchView<Scalar> view);

  cublasHandle_t cublas_ = nullptr;
  cudaStream_t stream_ = nullptr;
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t graph_exec_ = nullptr;
  bool graph_valid_ = false;

  CudaDevicePtr<Scalar> d_centroids_;
  CudaDevicePtr<Scalar> d_centroid_norms_;
  CudaDevicePtr<Scalar> d_embedding_;
  CudaDevicePtr<Scalar> d_embed_norm_;
  CudaDevicePtr<Scalar> d_dots_;
  CudaDevicePtr<int> d_best_idx_;
  CudaDevicePtr<Scalar> d_best_dist_;

  CudaPinnedPtr<Scalar> h_embedding_;
  CudaPinnedPtr<int> h_best_idx_;
  CudaPinnedPtr<Scalar> h_best_dist_;

  static constexpr int kNumPipelineStages = 2;

  struct PipelineStage {
    cudaStream_t stream = nullptr;
    cudaEvent_t event = nullptr;
    CudaDevicePtr<Scalar> d_queries;
    CudaDevicePtr<Scalar> d_norms;
    CudaDevicePtr<Scalar> d_dots;
    CudaDevicePtr<int> d_idx;
    CudaDevicePtr<Scalar> d_dist;
    CudaPinnedPtr<Scalar> h_queries;
    CudaPinnedPtr<int> h_idx;
    CudaPinnedPtr<Scalar> h_dist;
    int capacity = 0;
  };

  PipelineStage stages_[kNumPipelineStages];
  cublasHandle_t pipeline_cublas_[kNumPipelineStages] = {};
  bool pipeline_initialized_ = false;

  void init_pipeline();
  void ensure_stage_capacity(int stage_idx, int count);

  int n_clusters_ = 0;
  int dim_ = 0;
};

#endif  // NORDLYS_HAS_CUDA

// =============================================================================
// Factory & Engine
// =============================================================================

[[nodiscard]] inline bool cuda_available() noexcept {
#ifdef NORDLYS_HAS_CUDA
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  return err == cudaSuccess && device_count > 0;
#else
  return false;
#endif
}

template <typename Scalar>
[[nodiscard]] std::unique_ptr<IClusterBackend<Scalar>> create_backend(Device device) {
  return std::visit(overloaded{
      [](CpuDevice) -> std::unique_ptr<IClusterBackend<Scalar>> {
#ifndef __CUDACC__
          return std::make_unique<CpuClusterBackend<Scalar>>();
#else
          throw std::runtime_error("CPU backend not available in CUDA compilation");
#endif
      },
      [](CudaDevice) -> std::unique_ptr<IClusterBackend<Scalar>> {
#ifdef NORDLYS_HAS_CUDA
          if (cuda_available()) {
              return std::make_unique<CudaClusterBackend<Scalar>>();
          }
          throw std::runtime_error("CUDA not available");
#else
          throw std::runtime_error("CUDA backend not compiled");
#endif
      }
  }, device);
}
