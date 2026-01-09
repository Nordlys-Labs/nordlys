#pragma once
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

template <typename T> class Matrix {
public:
  using value_type = T;
  using Scalar = T;

  Matrix() = default;

  Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols), data_(rows * cols) {}

  Matrix(size_t rows, size_t cols, const T* src)
      : rows_(rows), cols_(cols), data_(src, src + rows * cols) {}

  [[nodiscard]] size_t rows() const noexcept { return rows_; }
  [[nodiscard]] size_t cols() const noexcept { return cols_; }
  [[nodiscard]] size_t size() const noexcept { return data_.size(); }

  [[nodiscard]] T* data() noexcept { return data_.data(); }
  [[nodiscard]] const T* data() const noexcept { return data_.data(); }

  T& operator()(size_t row, size_t col) { return data_[row * cols_ + col]; }
  const T& operator()(size_t row, size_t col) const { return data_[row * cols_ + col]; }

  void resize(size_t rows, size_t cols) {
    rows_ = rows;
    cols_ = cols;
    data_.resize(rows * cols);
  }

private:
  size_t rows_ = 0;
  size_t cols_ = 0;
  std::vector<T> data_;
};

template <typename Scalar> using EmbeddingMatrixT = Matrix<Scalar>;
