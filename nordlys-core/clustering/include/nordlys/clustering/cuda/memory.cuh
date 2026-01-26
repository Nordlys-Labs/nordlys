#pragma once
#ifdef NORDLYS_HAS_CUDA

#  include <cstddef>
#  include <nordlys/clustering/cuda/common.cuh>
#  include <utility>

template <typename T> class CudaDevicePtr {
public:
  CudaDevicePtr() = default;

  explicit CudaDevicePtr(size_t count) : count_(count) {
    if (count > 0) {
      NORDLYS_CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
    }
  }

  ~CudaDevicePtr() { free(); }

  CudaDevicePtr(CudaDevicePtr&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
    other.ptr_ = nullptr;
    other.count_ = 0;
  }

  CudaDevicePtr& operator=(CudaDevicePtr&& other) noexcept {
    if (this != &other) {
      free();
      ptr_ = other.ptr_;
      count_ = other.count_;
      other.ptr_ = nullptr;
      other.count_ = 0;
    }
    return *this;
  }

  CudaDevicePtr(const CudaDevicePtr&) = delete;
  CudaDevicePtr& operator=(const CudaDevicePtr&) = delete;

  T* get() const { return ptr_; }
  size_t size() const { return count_; }
  bool empty() const { return ptr_ == nullptr; }

  void reset(size_t count = 0) {
    free();
    count_ = count;
    if (count > 0) {
      NORDLYS_CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
    }
  }

private:
  void free() {
    if (ptr_) {
      cudaFree(ptr_);
      ptr_ = nullptr;
      count_ = 0;
    }
  }

  T* ptr_ = nullptr;
  size_t count_ = 0;
};

template <typename T> class CudaPinnedPtr {
public:
  CudaPinnedPtr() = default;

  explicit CudaPinnedPtr(size_t count) : count_(count) {
    if (count > 0) {
      NORDLYS_CUDA_CHECK(cudaMallocHost(&ptr_, count * sizeof(T)));
    }
  }

  ~CudaPinnedPtr() { free(); }

  CudaPinnedPtr(CudaPinnedPtr&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
    other.ptr_ = nullptr;
    other.count_ = 0;
  }

  CudaPinnedPtr& operator=(CudaPinnedPtr&& other) noexcept {
    if (this != &other) {
      free();
      ptr_ = other.ptr_;
      count_ = other.count_;
      other.ptr_ = nullptr;
      other.count_ = 0;
    }
    return *this;
  }

  CudaPinnedPtr(const CudaPinnedPtr&) = delete;
  CudaPinnedPtr& operator=(const CudaPinnedPtr&) = delete;

  T* get() const { return ptr_; }
  size_t size() const { return count_; }
  bool empty() const { return ptr_ == nullptr; }

  void reset(size_t count = 0) {
    free();
    count_ = count;
    if (count > 0) {
      NORDLYS_CUDA_CHECK(cudaMallocHost(&ptr_, count * sizeof(T)));
    }
  }

private:
  void free() {
    if (ptr_) {
      cudaFreeHost(ptr_);
      ptr_ = nullptr;
      count_ = 0;
    }
  }

  T* ptr_ = nullptr;
  size_t count_ = 0;
};

#endif
