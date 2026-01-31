#pragma once
#ifdef NORDLYS_HAS_CUDA

#include <cstddef>
#include <iostream>
#include <utility>

#include <nordlys/clustering/cuda/common.cuh>

namespace nordlys::clustering::cuda {

template <typename T>
class DevicePtr {
public:
  DevicePtr() = default;

  explicit DevicePtr(size_t count, cudaStream_t stream = cudaStreamPerThread) : count_(count) {
    if (count > 0) {
      NORDLYS_CUDA_CHECK(cudaMallocAsync(&ptr_, count * sizeof(T), stream));
    }
  }

  ~DevicePtr() noexcept {
    if (ptr_) {
      // Don't throw in destructor - just log error if free fails
      auto err = cudaFree(ptr_);
      if (err != cudaSuccess) [[unlikely]] {
        std::cerr << "DevicePtr::~DevicePtr() cudaFree failed: " 
                  << cudaGetErrorString(err) << "\n";
      }
      ptr_ = nullptr;
      count_ = 0;
    }
  }

  DevicePtr(DevicePtr&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
    other.ptr_ = nullptr;
    other.count_ = 0;
  }

  auto operator=(DevicePtr&& other) noexcept -> DevicePtr& {
    if (this != &other) {
      free_async(nullptr);  // Use async free for consistency
      ptr_ = other.ptr_;
      count_ = other.count_;
      other.ptr_ = nullptr;
      other.count_ = 0;
    }
    return *this;
  }

  DevicePtr(const DevicePtr&) = delete;
  auto operator=(const DevicePtr&) = delete;

  [[nodiscard]] auto get() const noexcept -> T* { return ptr_; }
  [[nodiscard]] auto size() const noexcept -> size_t { return count_; }
  [[nodiscard]] auto empty() const noexcept -> bool { return ptr_ == nullptr; }

  void reset(size_t count = 0, cudaStream_t stream = cudaStreamPerThread) {
    free_async(stream);
    count_ = count;
    if (count > 0) {
      NORDLYS_CUDA_CHECK(cudaMallocAsync(&ptr_, count * sizeof(T), stream));
    }
  }

  void free_async(cudaStream_t stream) noexcept {
    if (ptr_) {
      auto err = cudaFreeAsync(ptr_, stream);
      if (err != cudaSuccess) [[unlikely]] {
        std::cerr << "DevicePtr::free_async() failed: " 
                  << cudaGetErrorString(err) << "\n";
      }
      ptr_ = nullptr;
      count_ = 0;
    }
  }

private:
  T* ptr_ = nullptr;
  size_t count_ = 0;
};

template <typename T>
class PinnedPtr {
public:
  PinnedPtr() = default;

  explicit PinnedPtr(size_t count) : count_(count) {
    if (count > 0) {
      NORDLYS_CUDA_CHECK(cudaMallocHost(&ptr_, count * sizeof(T)));
    }
  }

  ~PinnedPtr() noexcept {
    if (ptr_) {
      auto err = cudaFreeHost(ptr_);
      if (err != cudaSuccess) [[unlikely]] {
        std::cerr << "PinnedPtr::~PinnedPtr() cudaFreeHost failed: " 
                  << cudaGetErrorString(err) << "\n";
      }
      ptr_ = nullptr;
      count_ = 0;
    }
  }

  PinnedPtr(PinnedPtr&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
    other.ptr_ = nullptr;
    other.count_ = 0;
  }

  auto operator=(PinnedPtr&& other) noexcept -> PinnedPtr& {
    if (this != &other) {
      free();
      ptr_ = other.ptr_;
      count_ = other.count_;
      other.ptr_ = nullptr;
      other.count_ = 0;
    }
    return *this;
  }

  PinnedPtr(const PinnedPtr&) = delete;
  auto operator=(const PinnedPtr&) = delete;

  [[nodiscard]] auto get() const noexcept -> T* { return ptr_; }
  [[nodiscard]] auto size() const noexcept -> size_t { return count_; }
  [[nodiscard]] auto empty() const noexcept -> bool { return ptr_ == nullptr; }

  void reset(size_t count = 0) {
    free();
    count_ = count;
    if (count > 0) {
      NORDLYS_CUDA_CHECK(cudaMallocHost(&ptr_, count * sizeof(T)));
    }
  }

private:
  void free() noexcept {
    if (ptr_) {
      auto err = cudaFreeHost(ptr_);
      if (err != cudaSuccess) [[unlikely]] {
        std::cerr << "PinnedPtr::free() cudaFreeHost failed: " 
                  << cudaGetErrorString(err) << "\n";
      }
      ptr_ = nullptr;
      count_ = 0;
    }
  }

  T* ptr_ = nullptr;
  size_t count_ = 0;
};

} // namespace nordlys::clustering::cuda

#endif
