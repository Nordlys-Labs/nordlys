#pragma once
#include <Eigen/Dense>
#include <memory>
#include <utility>

#include "nordlys_core/cluster_backend.hpp"

// Templated types for multi-precision support
template<typename Scalar>
using EmbeddingVectorT = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

// Row-major storage to match serialized binary format
template<typename Scalar>
using EmbeddingMatrixT = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template<typename Scalar = float>
class ClusterEngineT {
public:
  ClusterEngineT()
      : backend_(create_cluster_backend<Scalar>(ClusterBackendType::Auto)) {}

  explicit ClusterEngineT(ClusterBackendType backend_type)
      : backend_(create_cluster_backend<Scalar>(backend_type)) {}

  ~ClusterEngineT() = default;

  ClusterEngineT(ClusterEngineT&&) noexcept = default;
  ClusterEngineT& operator=(ClusterEngineT&&) noexcept = default;
  ClusterEngineT(const ClusterEngineT&) = delete;
  ClusterEngineT& operator=(const ClusterEngineT&) = delete;

  void load_centroids(const EmbeddingMatrixT<Scalar>& centers) {
    dim_ = static_cast<int>(centers.cols());
    int n_clusters = static_cast<int>(centers.rows());

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> row_major = centers;
    backend_->load_centroids(row_major.data(), n_clusters, dim_);
  }

  [[nodiscard]] std::pair<int, Scalar> assign(const EmbeddingVectorT<Scalar>& embedding) {
    return backend_->assign(embedding.data(), static_cast<int>(embedding.size()));
  }

  [[nodiscard]] int get_n_clusters() const noexcept {
    return backend_->get_n_clusters();
  }

  [[nodiscard]] bool is_gpu_accelerated() const noexcept {
    return backend_->is_gpu_accelerated();
  }

private:
  std::unique_ptr<IClusterBackendT<Scalar>> backend_;
  int dim_ = 0;
};
