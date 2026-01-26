#pragma once

#include <nordlys/clustering/cluster.hpp>
#include <usearch/index.hpp>
#include <usearch/index_dense.hpp>
#include <vector>

class CpuClusterBackend : public IClusterBackend {
public:
  void load_centroids(const float* data, size_t n_clusters, size_t dim) override;

  [[nodiscard]] std::pair<int, float> assign(EmbeddingView view) override;

  [[nodiscard]] std::vector<std::pair<int, float>> assign_batch(EmbeddingBatchView view) override;

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
