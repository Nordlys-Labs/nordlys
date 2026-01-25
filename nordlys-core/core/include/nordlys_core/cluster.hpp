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

#include "matrix.hpp"

enum class ClusterBackendType { Cpu, CUDA };

template <typename Scalar> class IClusterBackend {
public:
  virtual ~IClusterBackend() = default;

  IClusterBackend(const IClusterBackend&) = delete;
  IClusterBackend& operator=(const IClusterBackend&) = delete;
  IClusterBackend(IClusterBackend&&) = delete;
  IClusterBackend& operator=(IClusterBackend&&) = delete;

  virtual void load_centroids(const Scalar* data, int n_clusters, int dim) = 0;

  [[nodiscard]] virtual std::pair<int, Scalar> assign(const Scalar* embedding, int dim) = 0;

  [[nodiscard]] virtual std::vector<std::pair<int, Scalar>> assign_batch(const Scalar* embeddings,
                                                                         int count, int dim) {
    if (count < 0 || dim <= 0) [[unlikely]] {
      throw std::invalid_argument("count must be non-negative and dim must be positive");
    }
    std::vector<std::pair<int, Scalar>> results;
    results.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
      results.push_back(assign(embeddings + i * dim, dim));
    }
    return results;
  }

  [[nodiscard]] virtual int get_n_clusters() const noexcept = 0;
  [[nodiscard]] virtual int get_dim() const noexcept = 0;
  [[nodiscard]] virtual bool is_gpu_accelerated() const noexcept = 0;

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
  void load_centroids(const Scalar* data, int n_clusters, int dim) override {
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

    n_clusters_ = n_clusters;
    dim_ = dim;

    using namespace unum::usearch;
    auto scalar_kind = std::is_same_v<Scalar, float> ? scalar_kind_t::f32_k : scalar_kind_t::f64_k;
    metric_ = metric_punned_t(d, metric_kind_t::l2sq_k, scalar_kind);

    centroids_.resize(total_size);
    std::memcpy(centroids_.data(), data, centroids_.size() * sizeof(Scalar));
  }

  [[nodiscard]] std::pair<int, Scalar> assign(const Scalar* embedding, int dim) override {
    if (n_clusters_ == 0) return {-1, Scalar{0}};
    if (dim != dim_) [[unlikely]] {
      throw std::invalid_argument("dimension mismatch in assign");
    }

    const auto* emb_bytes = reinterpret_cast<const unum::usearch::byte_t*>(embedding);

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

  [[nodiscard]] std::vector<std::pair<int, Scalar>> assign_batch(const Scalar* embeddings,
                                                                 int count, int dim) override {
    if (count < 0) [[unlikely]] {
      throw std::invalid_argument("count must be non-negative");
    }
    if (n_clusters_ > 0 && dim != dim_) [[unlikely]] {
      throw std::invalid_argument("dimension mismatch in assign_batch");
    }

    std::vector<std::pair<int, Scalar>> results(static_cast<size_t>(count));

#  ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#  endif
    for (int i = 0; i < count; ++i) {
      results[static_cast<size_t>(i)] = assign(embeddings + i * dim_, dim_);
    }

    return results;
  }

  [[nodiscard]] int get_n_clusters() const noexcept override { return n_clusters_; }
  [[nodiscard]] int get_dim() const noexcept override { return dim_; }
  [[nodiscard]] bool is_gpu_accelerated() const noexcept override { return false; }

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

  void load_centroids(const Scalar* data, int n_clusters, int dim) override;

  [[nodiscard]] std::pair<int, Scalar> assign(const Scalar* embedding, int dim) override;

  [[nodiscard]] std::vector<std::pair<int, Scalar>> assign_batch(const Scalar* embeddings,
                                                                 int count, int dim) override;

  [[nodiscard]] int get_n_clusters() const noexcept override { return n_clusters_; }
  [[nodiscard]] int get_dim() const noexcept override { return dim_; }
  [[nodiscard]] bool is_gpu_accelerated() const noexcept override { return true; }

private:
  void free_memory();
  void capture_graph();
  void ensure_batch_capacity(int count);

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
[[nodiscard]] std::unique_ptr<IClusterBackend<Scalar>> create_cluster_backend(
    ClusterBackendType type) {
  switch (type) {
    case ClusterBackendType::Cpu:
#ifndef __CUDACC__
      return std::make_unique<CpuClusterBackend<Scalar>>();
#else
      break;
#endif

    case ClusterBackendType::CUDA:
#ifdef NORDLYS_HAS_CUDA
      if (cuda_available()) {
        return std::make_unique<CudaClusterBackend<Scalar>>();
      }
#endif
#ifndef __CUDACC__
      return std::make_unique<CpuClusterBackend<Scalar>>();
#else
      break;
#endif
  }

#ifndef __CUDACC__
  return std::make_unique<CpuClusterBackend<Scalar>>();
#else
  return nullptr;
#endif
}

template <typename Scalar = float> class ClusterEngine {
public:
  ClusterEngine() : device_(ClusterBackendType::Cpu), backend_(create_cluster_backend<Scalar>(ClusterBackendType::Cpu)) {
    if (!backend_) [[unlikely]] {
      throw std::runtime_error("failed to create cluster backend");
    }
  }

  explicit ClusterEngine(ClusterBackendType type) : device_(type), backend_(create_cluster_backend<Scalar>(type)) {
    if (!backend_) [[unlikely]] {
      throw std::runtime_error("failed to create cluster backend");
    }
  }

  void load_centroids(const EmbeddingMatrix<Scalar>& centers) {
    if (centers.rows() > static_cast<size_t>(std::numeric_limits<int>::max())
        || centers.cols() > static_cast<size_t>(std::numeric_limits<int>::max())) [[unlikely]] {
      throw std::invalid_argument("matrix dimensions exceed int range");
    }
    backend_->load_centroids(centers.data(), static_cast<int>(centers.rows()),
                             static_cast<int>(centers.cols()));
  }

  [[nodiscard]] auto assign(const Scalar* embedding, size_t dim) {
    if (dim > static_cast<size_t>(std::numeric_limits<int>::max())) [[unlikely]] {
      throw std::invalid_argument("dim exceeds int range");
    }
    int backend_dim = backend_->get_dim();
    if (backend_dim > 0 && static_cast<int>(dim) != backend_dim) [[unlikely]] {
      throw std::invalid_argument("dimension mismatch: expected " + std::to_string(backend_dim)
                                  + ", got " + std::to_string(dim));
    }
    return backend_->assign(embedding, static_cast<int>(dim));
  }

  [[nodiscard]] auto assign_batch(const Scalar* embeddings, size_t count, size_t dim) {
    if (count > static_cast<size_t>(std::numeric_limits<int>::max())
        || dim > static_cast<size_t>(std::numeric_limits<int>::max())) [[unlikely]] {
      throw std::invalid_argument("count or dim exceeds int range");
    }
    int backend_dim = backend_->get_dim();
    if (backend_dim > 0 && static_cast<int>(dim) != backend_dim) [[unlikely]] {
      throw std::invalid_argument("dimension mismatch: expected " + std::to_string(backend_dim)
                                  + ", got " + std::to_string(dim));
    }
    return backend_->assign_batch(embeddings, static_cast<int>(count), static_cast<int>(dim));
  }

  [[nodiscard]] int get_n_clusters() const noexcept { return backend_->get_n_clusters(); }
  [[nodiscard]] ClusterBackendType get_device() const noexcept { return device_; }
  [[nodiscard]] bool is_gpu_accelerated() const noexcept { return backend_->is_gpu_accelerated(); }

private:
  ClusterBackendType device_;
  std::unique_ptr<IClusterBackend<Scalar>> backend_;
};
