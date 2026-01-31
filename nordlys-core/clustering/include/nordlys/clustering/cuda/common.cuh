#pragma once
#ifdef NORDLYS_HAS_CUDA

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <concepts>
#include <stdexcept>
#include <string>

namespace nordlys::clustering::cuda {

#define NORDLYS_CUDA_CHECK(call)                                                               \
  do {                                                                                         \
    cudaError_t err__ = (call);                                                                \
    if (err__ != cudaSuccess) [[unlikely]] {                                                   \
      throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":"                  \
                               + std::to_string(__LINE__) + ": " + cudaGetErrorString(err__)); \
    }                                                                                          \
  } while (0)

#define NORDLYS_CUBLAS_CHECK(call)                                                               \
  do {                                                                                           \
    cublasStatus_t stat__ = (call);                                                              \
    if (stat__ != CUBLAS_STATUS_SUCCESS) [[unlikely]] {                                          \
      throw std::runtime_error(std::string("cuBLAS error at ") + __FILE__ + ":"                  \
                               + std::to_string(__LINE__) + ": code " + std::to_string(stat__)); \
    }                                                                                            \
  } while (0)

template <typename T>
struct TypeTraits;

template <>
struct TypeTraits<float> {
  static constexpr float max_value = FLT_MAX;
  static constexpr float min_value = -FLT_MAX;
  static constexpr float epsilon = FLT_EPSILON;
};

template <>
struct TypeTraits<double> {
  static constexpr double max_value = DBL_MAX;
  static constexpr double min_value = -DBL_MAX;
  static constexpr double epsilon = DBL_EPSILON;
};

template <std::floating_point T>
inline constexpr TypeTraits<T> type_traits{};

} // namespace nordlys::clustering::cuda

#endif
