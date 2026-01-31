#ifdef NORDLYS_HAS_CUDA

#include <gtest/gtest.h>

#include <nordlys/clustering/cuda/config.cuh>
#include <nordlys/clustering/cuda/distance.cuh>

namespace nordlys::clustering::cuda::test {

class ConfigTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(0));
  }
};

TEST_F(ConfigTest, GetOptimalConfigBasic) {
  auto config = get_optimal_config(
    cuda::fused_l2_argmin_with_norm<float>,
    1024,
    cuda::fused_l2_argmin_shared_mem_size<float>()
  );
  
  EXPECT_GT(config.block_size, 0);
  EXPECT_LE(config.block_size, 1024);
  EXPECT_GT(config.grid_size, 0);
}

TEST_F(ConfigTest, GetOptimalConfigWithSharedMem) {
  size_t smem = 4096;
  auto config = get_optimal_config(
    cuda::fused_l2_argmin_with_norm<float>,
    512,
    smem
  );
  
  EXPECT_EQ(config.shared_mem, smem);
}

TEST_F(ConfigTest, SharedMemSizeCompilation) {
  constexpr auto smem_float = cuda::fused_l2_argmin_shared_mem_size<float>();
  constexpr auto smem_double = cuda::fused_l2_argmin_shared_mem_size<double>();
  
  static_assert(smem_float > 0, "Shared mem size must be positive");
  static_assert(smem_double > smem_float, "Double should need more shared mem");
}

} // namespace nordlys::clustering::cuda::test

#endif // NORDLYS_HAS_CUDA
