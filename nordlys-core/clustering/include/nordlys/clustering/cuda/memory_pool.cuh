#pragma once
#ifdef NORDLYS_HAS_CUDA

#include <cuda_runtime.h>
#include <nordlys/clustering/cuda/common.cuh>

namespace nordlys::clustering::cuda {

class MemoryPool {
public:
  MemoryPool() {
    int device;
    NORDLYS_CUDA_CHECK(cudaGetDevice(&device));
    device_ = device;
    
    // Get default memory pool for the device
    NORDLYS_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool_, device_));
    
    // Configure memory pool for better performance
    // Release threshold: keep up to 1GB of memory cached
    uint64_t threshold = 1ULL << 30;  // 1GB
    NORDLYS_CUDA_CHECK(cudaMemPoolSetAttribute(
      pool_, cudaMemPoolAttrReleaseThreshold, &threshold));
    
    // Enable reuse across streams
    int enable = 1;
    NORDLYS_CUDA_CHECK(cudaMemPoolSetAttribute(
      pool_, cudaMemPoolReuseFollowEventDependencies, &enable));
  }
  
  [[nodiscard]] auto get() const noexcept -> cudaMemPool_t {
    return pool_;
  }
  
  [[nodiscard]] auto device() const noexcept -> int {
    return device_;
  }
  
  // Trim excess memory back to OS
  void trim_to(uint64_t min_bytes_to_keep = 0) {
    NORDLYS_CUDA_CHECK(cudaMemPoolTrimTo(pool_, min_bytes_to_keep));
  }
  
  // Get current pool stats
  struct Stats {
    uint64_t reserved_bytes;
    uint64_t used_bytes;
  };
  
  [[nodiscard]] auto stats() const -> Stats {
    uint64_t reserved = 0;
    uint64_t used = 0;
    
    NORDLYS_CUDA_CHECK(cudaMemPoolGetAttribute(
      pool_, cudaMemPoolAttrReservedMemCurrent, &reserved));
    NORDLYS_CUDA_CHECK(cudaMemPoolGetAttribute(
      pool_, cudaMemPoolAttrUsedMemCurrent, &used));
    
    return Stats{reserved, used};
  }
  
private:
  cudaMemPool_t pool_ = nullptr;
  int device_ = 0;
};

// Global memory pool instance (one per device)
inline auto get_memory_pool() -> MemoryPool& {
  static thread_local MemoryPool pool;
  return pool;
}

} // namespace nordlys::clustering::cuda

#endif
