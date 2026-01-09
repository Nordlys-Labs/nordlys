#pragma once
#include <memory>
#include <utility>

#include "nordlys_core/cluster_backend.hpp"
#include "nordlys_core/matrix.hpp"
#include "nordlys_core/tracy.hpp"

template <typename Scalar = float> class ClusterEngineT {
public:
  ClusterEngineT() : backend_(create_cluster_backend<Scalar>(ClusterBackendType::Auto)) {}

  explicit ClusterEngineT(ClusterBackendType backend_type)
      : backend_(create_cluster_backend<Scalar>(backend_type)) {}

  ~ClusterEngineT() = default;

  ClusterEngineT(ClusterEngineT&&) noexcept = default;
  ClusterEngineT& operator=(ClusterEngineT&&) noexcept = default;
  ClusterEngineT(const ClusterEngineT&) = delete;
  ClusterEngineT& operator=(const ClusterEngineT&) = delete;

  void load_centroids(const EmbeddingMatrixT<Scalar>& centers) {
    NORDLYS_ZONE;
    dim_ = static_cast<int>(centers.cols());
    int n_clusters = static_cast<int>(centers.rows());
    backend_->load_centroids(centers.data(), n_clusters, dim_);
  }

  [[nodiscard]] std::pair<int, Scalar> assign(const Scalar* embedding, size_t size) {
    NORDLYS_ZONE;
    return backend_->assign(embedding, static_cast<int>(size));
  }

  [[nodiscard]] int get_n_clusters() const noexcept { return backend_->get_n_clusters(); }

  [[nodiscard]] bool is_gpu_accelerated() const noexcept { return backend_->is_gpu_accelerated(); }

private:
  std::unique_ptr<IClusterBackendT<Scalar>> backend_;
  int dim_ = 0;
};
