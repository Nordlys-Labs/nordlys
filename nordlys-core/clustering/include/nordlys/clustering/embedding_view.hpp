#pragma once

#include <cstddef>
#include <span>

#include <nordlys/common/device.hpp>

namespace nordlys::clustering {

struct EmbeddingView {
  const float* data;
  size_t dim;
  Device device;

  [[nodiscard]] constexpr auto empty() const noexcept -> bool { return data == nullptr || dim == 0; }
  [[nodiscard]] constexpr auto size_bytes() const noexcept -> size_t { return dim * sizeof(float); }

  [[nodiscard]] auto span() const noexcept -> std::span<const float> {
    return std::span<const float>(data, dim);
  }
};

struct EmbeddingBatchView {
  const float* data;
  size_t count;
  size_t dim;
  Device device;

  [[nodiscard]] constexpr auto empty() const noexcept -> bool { return data == nullptr || count == 0; }
  [[nodiscard]] constexpr auto total_elements() const noexcept -> size_t { return count * dim; }
  [[nodiscard]] constexpr auto size_bytes() const noexcept -> size_t { return total_elements() * sizeof(float); }

  [[nodiscard]] auto span() const noexcept -> std::span<const float> {
    return std::span<const float>(data, total_elements());
  }
};

} // namespace nordlys::clustering
