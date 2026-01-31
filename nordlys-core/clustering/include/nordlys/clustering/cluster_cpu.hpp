#pragma once

#include <nordlys/clustering/cluster.hpp>
#include <usearch/index.hpp>
#include <usearch/index_dense.hpp>
#include <vector>

namespace nordlys::clustering {

class CpuClusterBackend : public IClusterBackend {
public:
  void load_centroids(const float* data, size_t n_clusters, size_t dim) override;

  [[nodiscard]] auto assign(EmbeddingView view) -> std::pair<int, float> override;

  [[nodiscard]] auto assign_batch(EmbeddingBatchView view)
    -> std::vector<std::pair<int, float>> override;

  [[nodiscard]] auto n_clusters() const noexcept -> size_t override {
    return static_cast<size_t>(n_clusters_);
  }

  [[nodiscard]] auto dim() const noexcept -> size_t override { 
    return static_cast<size_t>(dim_); 
  }

private:
  std::vector<float> centroids_;
  unum::usearch::metric_punned_t metric_;
  int n_clusters_ = 0;
  int dim_ = 0;
};

} // namespace nordlys::clustering
