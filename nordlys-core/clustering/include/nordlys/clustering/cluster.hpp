#pragma once

#include <memory>
#include <nordlys/clustering/embedding_view.hpp>
#include <nordlys/common/device.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

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

  [[nodiscard]] virtual std::pair<int, float> assign(EmbeddingView view) = 0;

  [[nodiscard]] virtual std::vector<std::pair<int, float>> assign_batch(EmbeddingBatchView view)
      = 0;

  [[nodiscard]] virtual size_t n_clusters() const = 0;
  [[nodiscard]] virtual size_t dim() const = 0;

protected:
  IClusterBackend() = default;
};

// =============================================================================
// Factory Functions
// =============================================================================

[[nodiscard]] bool cuda_available() noexcept;
[[nodiscard]] std::unique_ptr<IClusterBackend> create_backend(Device device);
