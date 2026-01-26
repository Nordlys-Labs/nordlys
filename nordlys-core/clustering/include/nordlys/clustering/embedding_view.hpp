#pragma once

#include <cstddef>

#include <nordlys/common/device.hpp>

struct EmbeddingView {
  const float* data;
  size_t dim;
  Device device;
};

struct EmbeddingBatchView {
  const float* data;
  size_t count;
  size_t dim;
  Device device;
};
