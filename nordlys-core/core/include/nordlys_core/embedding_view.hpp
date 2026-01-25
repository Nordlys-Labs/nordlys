#pragma once

#include "device.hpp"

template <typename Scalar>
struct EmbeddingView {
    const Scalar* data;
    size_t dim;
    Device device;
};

template <typename Scalar>
struct EmbeddingBatchView {
    const Scalar* data;
    size_t count;
    size_t dim;
    Device device;
};
