#pragma once
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

// Forward declaration
template <typename T, typename E> class Result;

/// Helper class to create error result (similar to std::unexpected)
template <typename E> class Unexpected {
public:
  explicit constexpr Unexpected(const E& err) : error_(err) {}
  explicit constexpr Unexpected(E&& err) : error_(std::move(err)) {}

  constexpr const E& error() const& noexcept { return error_; }
  constexpr E& error() & noexcept { return error_; }
  constexpr E&& error() && noexcept { return std::move(error_); }

private:
  E error_;
};

// Deduction guide
template <typename E> Unexpected(E) -> Unexpected<E>;

/// C++20-compatible Result<T, E> type as replacement for std::expected
/// Provides similar API to std::expected for error handling
template <typename T, typename E> class Result {
public:
  // Constructors for success value
  constexpr Result(const T& value) : storage_(value) {}
  constexpr Result(T&& value) : storage_(std::move(value)) {}

  // Constructor from Unexpected (for error)
  template <typename U> constexpr Result(const Unexpected<U>& err) : storage_(E(err.error())) {}

  template <typename U> constexpr Result(Unexpected<U>&& err)
      : storage_(E(std::move(err).error())) {}

  // Check if result contains a value
  [[nodiscard]] constexpr bool has_value() const noexcept {
    return std::holds_alternative<T>(storage_);
  }

  [[nodiscard]] constexpr explicit operator bool() const noexcept { return has_value(); }

  // Access value (throws if error)
  [[nodiscard]] constexpr T& value() & {
    if (!has_value()) {
      throw std::runtime_error("Result contains error");
    }
    return std::get<T>(storage_);
  }

  [[nodiscard]] constexpr const T& value() const& {
    if (!has_value()) {
      throw std::runtime_error("Result contains error");
    }
    return std::get<T>(storage_);
  }

  [[nodiscard]] constexpr T&& value() && {
    if (!has_value()) {
      throw std::runtime_error("Result contains error");
    }
    return std::get<T>(std::move(storage_));
  }

  // Access error (throws if value)
  [[nodiscard]] constexpr E& error() & {
    if (has_value()) {
      throw std::runtime_error("Result contains value");
    }
    return std::get<E>(storage_);
  }

  [[nodiscard]] constexpr const E& error() const& {
    if (has_value()) {
      throw std::runtime_error("Result contains value");
    }
    return std::get<E>(storage_);
  }

  // Operators for convenient access
  [[nodiscard]] constexpr T& operator*() & { return value(); }
  [[nodiscard]] constexpr const T& operator*() const& { return value(); }
  [[nodiscard]] constexpr T&& operator*() && { return std::move(value()); }

  [[nodiscard]] constexpr T* operator->() { return &value(); }
  [[nodiscard]] constexpr const T* operator->() const { return &value(); }

private:
  std::variant<T, E> storage_;
};
