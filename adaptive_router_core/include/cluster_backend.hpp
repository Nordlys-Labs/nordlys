#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>

// Backend type enumeration
enum class ClusterBackendType { CPU, CUDA, Auto };

// Templated abstract interface for cluster assignment backends
template<typename Scalar>
class IClusterBackendT {
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

// CPU backend using Eigen (header-only implementation)
template<typename Scalar>
class CpuClusterBackendT : public IClusterBackendT<Scalar> {
public:
  CpuClusterBackendT() = default;
  ~CpuClusterBackendT() override = default;

  void load_centroids(const Scalar* data, int n_clusters, int dim) override {
    n_clusters_ = n_clusters;
    dim_ = dim;

    // Map raw data to Eigen matrix (row-major input)
    Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> input(
        data, n_clusters, dim);
    centroids_ = input;
  }

  [[nodiscard]] std::pair<int, Scalar> assign(const Scalar* embedding, int dim) override {
    if (n_clusters_ == 0 || dim != dim_) {
      return {-1, Scalar(0)};
    }

    // Map embedding to Eigen vector (zero-copy)
    Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> emb(embedding, dim);

    // Find nearest centroid using vectorized operations
    int best_cluster = 0;
    Scalar best_distance = std::numeric_limits<Scalar>::max();

    for (int i = 0; i < n_clusters_; ++i) {
      Scalar dist = (centroids_.row(i).transpose() - emb).squaredNorm();
      if (dist < best_distance) {
        best_distance = dist;
        best_cluster = i;
      }
    }

    return {best_cluster, std::sqrt(best_distance)};
  }

  [[nodiscard]] int get_n_clusters() const noexcept override { return n_clusters_; }

  [[nodiscard]] int get_dim() const noexcept override { return dim_; }

  [[nodiscard]] bool is_gpu_accelerated() const noexcept override { return false; }

private:
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> centroids_;
  int n_clusters_ = 0;
  int dim_ = 0;
};

// Factory function to create appropriate backend
template<typename Scalar>
[[nodiscard]] std::unique_ptr<IClusterBackendT<Scalar>> create_cluster_backend(
    ClusterBackendType type = ClusterBackendType::Auto);

// Check if CUDA is available at runtime
[[nodiscard]] bool cuda_available() noexcept;
