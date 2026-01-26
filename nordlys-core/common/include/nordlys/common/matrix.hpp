#pragma once
#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

template <typename T> class Matrix {
public:
  using value_type = T;
  using Scalar = T;

  Matrix() = default;

  Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols) {
    if (cols != 0 && rows > std::numeric_limits<size_t>::max() / cols) [[unlikely]] {
      throw std::overflow_error("Matrix size overflow: rows * cols exceeds size_t max");
    }
    data_.resize(rows * cols);
  }

  Matrix(size_t rows, size_t cols, const T* src) : rows_(rows), cols_(cols) {
    if (cols != 0 && rows > std::numeric_limits<size_t>::max() / cols) [[unlikely]] {
      throw std::overflow_error("Matrix size overflow: rows * cols exceeds size_t max");
    }
    size_t total = rows * cols;
    data_ = std::vector<T>(src, src + total);
  }

  [[nodiscard]] constexpr size_t rows() const noexcept { return rows_; }
  [[nodiscard]] constexpr size_t cols() const noexcept { return cols_; }
  [[nodiscard]] constexpr size_t size() const noexcept { return data_.size(); }

  [[nodiscard]] T* data() noexcept { return data_.data(); }
  [[nodiscard]] const T* data() const noexcept { return data_.data(); }

  T& operator()(size_t row, size_t col) { return data_[row * cols_ + col]; }
  const T& operator()(size_t row, size_t col) const { return data_[row * cols_ + col]; }

  void resize(size_t rows, size_t cols) {
    if (cols != 0 && rows > std::numeric_limits<size_t>::max() / cols) [[unlikely]] {
      throw std::overflow_error("Matrix size overflow: rows * cols exceeds size_t max");
    }
    rows_ = rows;
    cols_ = cols;
    data_.resize(rows * cols);
  }

private:
  size_t rows_ = 0;
  size_t cols_ = 0;
  std::vector<T> data_;
};

template <typename Scalar> using EmbeddingMatrix = Matrix<Scalar>;
