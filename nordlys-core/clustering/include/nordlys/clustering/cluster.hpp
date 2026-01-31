#pragma once

#include <memory>
#include <nordlys/clustering/embedding_view.hpp>
#include <nordlys/common/device.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nordlys::clustering {

// =============================================================================
// IClusterBackend Interface
// =============================================================================

class IClusterBackend {
public:
  virtual ~IClusterBackend() = default;

  IClusterBackend(const IClusterBackend&) = delete;
  IClusterBackend& operator=(const IClusterBackend&) = delete;
  IClusterBackend(IClusterBackend&&) = delete;
  IClusterBackend& operator=(IClusterBackend&&) = delete;

  virtual void load_centroids(const float* data, size_t n_clusters, size_t dim) = 0;

  [[nodiscard]] virtual auto assign(EmbeddingView view) -> std::pair<int, float> = 0;

  [[nodiscard]] virtual auto assign_batch(EmbeddingBatchView view)
    -> std::vector<std::pair<int, float>> = 0;

  [[nodiscard]] virtual auto n_clusters() const noexcept -> size_t = 0;
  [[nodiscard]] virtual auto dim() const noexcept -> size_t = 0;

protected:
  IClusterBackend() = default;
};

// =============================================================================
// Factory Functions
// =============================================================================

[[nodiscard]] auto cuda_available() noexcept -> bool;
[[nodiscard]] auto create_backend(Device device) -> std::unique_ptr<IClusterBackend>;

} // namespace nordlys::clustering
