#pragma once
#ifdef NORDLYS_HAS_CUDA

#include <span>
#include <vector>
#include <concepts>

namespace nordlys::clustering::cuda {

template <std::floating_point T>
[[nodiscard]] auto to_col_major(std::span<const T> row_major, size_t rows, size_t cols) 
  -> std::vector<T> 
{
  std::vector<T> col_major(rows * cols);
  for (size_t r = 0; r < rows; ++r) {
    for (size_t c = 0; c < cols; ++c) {
      col_major[c * rows + r] = row_major[r * cols + c];
    }
  }
  return col_major;
}

template <std::floating_point T>
[[nodiscard]] auto compute_squared_norms(std::span<const T> data, size_t n, size_t dim) 
  -> std::vector<T> 
{
  std::vector<T> norms(n);
  for (size_t i = 0; i < n; ++i) {
    const T* row = data.data() + i * dim;
    T sum = T{0};
    for (size_t j = 0; j < dim; ++j) {
      sum += row[j] * row[j];
    }
    norms[i] = sum;
  }
  return norms;
}

} // namespace nordlys::clustering::cuda

#endif
