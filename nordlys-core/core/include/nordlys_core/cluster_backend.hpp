#pragma once
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

// Backend type enumeration
enum class ClusterBackendType {
  Cpu,   // CPU with SimSIMD acceleration (default)
  CUDA,  // GPU acceleration
  Auto   // Auto-select best available
};

// Templated abstract interface for cluster assignment backends
template <typename Scalar> class IClusterBackendT {
public:
  virtual ~IClusterBackendT() = default;

  // Non-copyable, non-movable (polymorphic base class)
  IClusterBackendT(const IClusterBackendT&) = delete;
  IClusterBackendT& operator=(const IClusterBackendT&) = delete;
  IClusterBackendT(IClusterBackendT&&) = delete;
  IClusterBackendT& operator=(IClusterBackendT&&) = delete;

  // Load cluster centroids (n_clusters x dim matrix in row-major order)
  virtual void load_centroids(const Scalar* data, int n_clusters, int dim) = 0;

  // Assign embedding to nearest cluster
  // Returns (cluster_id, distance) pair
  [[nodiscard]] virtual std::pair<int, Scalar> assign(const Scalar* embedding, int dim) = 0;

  // Get number of clusters
  [[nodiscard]] virtual int get_n_clusters() const noexcept = 0;

  // Get embedding dimension
  [[nodiscard]] virtual int get_dim() const noexcept = 0;

  // Check if backend is GPU-accelerated
  [[nodiscard]] virtual bool is_gpu_accelerated() const noexcept = 0;

protected:
  IClusterBackendT() = default;
};

// CPU backend only available when not compiling with nvcc
// (usearch headers are not CUDA-compatible)
#ifndef __CUDACC__

#include <usearch/index.hpp>
#include <usearch/index_dense.hpp>

#include "nordlys_core/tracy.hpp"

// CPU backend: Hardware-accelerated implementation using SimSIMD
// Uses direct metric computation (brute-force) with SIMD instructions
template <typename Scalar> class CpuClusterBackendT : public IClusterBackendT<Scalar> {
public:
  CpuClusterBackendT() = default;
  ~CpuClusterBackendT() override = default;

  void load_centroids(const Scalar* data, int n_clusters, int dim) override {
    NORDLYS_ZONE_N("CpuClusterBackend::load_centroids");

    n_clusters_ = n_clusters;
    dim_ = dim;

    using namespace unum::usearch;

    scalar_kind_t scalar_kind
        = std::is_same_v<Scalar, float> ? scalar_kind_t::f32_k : scalar_kind_t::f64_k;

    metric_ = metric_punned_t(static_cast<std::size_t>(dim), metric_kind_t::l2sq_k, scalar_kind);

    // Store centroids directly for brute-force search (no HNSW index overhead)
    centroids_.resize(static_cast<std::size_t>(n_clusters) * static_cast<std::size_t>(dim));
    std::memcpy(centroids_.data(), data, centroids_.size() * sizeof(Scalar));
  }

  [[nodiscard]] std::pair<int, Scalar> assign(const Scalar* embedding, int) override {
    NORDLYS_ZONE_N("CpuClusterBackend::assign");

    if (n_clusters_ == 0) {
      return {-1, static_cast<Scalar>(0)};
    }

    // Brute-force search using SimSIMD-accelerated metric
    int best_cluster = -1;
    Scalar best_squared_dist = std::numeric_limits<Scalar>::max();

    const auto* emb_bytes = reinterpret_cast<const unum::usearch::byte_t*>(embedding);

    for (int i = 0; i < n_clusters_; ++i) {
      const Scalar* centroid_ptr = centroids_.data() + (i * dim_);
      const auto* centroid_bytes = reinterpret_cast<const unum::usearch::byte_t*>(centroid_ptr);

      // metric_punned_t stores dim internally, takes only 2 args
      auto squared_dist = static_cast<Scalar>(metric_(emb_bytes, centroid_bytes));

      if (squared_dist < best_squared_dist) {
        best_squared_dist = squared_dist;
        best_cluster = i;
      }
    }

    return {best_cluster, std::sqrt(best_squared_dist)};
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

// Factory function to create appropriate backend
template <typename Scalar>
[[nodiscard]] std::unique_ptr<IClusterBackendT<Scalar>> create_cluster_backend(
    ClusterBackendType type = ClusterBackendType::Auto);

// Check if CUDA is available at runtime
[[nodiscard]] bool cuda_available() noexcept;
