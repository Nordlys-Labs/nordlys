#ifdef NORDLYS_HAS_CUDA

#include <gtest/gtest.h>
#include <vector>

#include <nordlys/clustering/cuda/memory.cuh>
#include <nordlys/clustering/cuda/utils.cuh>

namespace nordlys::clustering::cuda::test {

class UtilsTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(0));
  }
};

TEST_F(UtilsTest, ToColMajor2x3) {
  std::vector<float> row = {1, 2, 3, 4, 5, 6};  // 2 rows, 3 cols
  auto col = cuda::to_col_major<float>(std::span<const float>(row), 2, 3);
  
  ASSERT_EQ(col.size(), 6);
  EXPECT_EQ(col[0], 1);
  EXPECT_EQ(col[1], 4);
  EXPECT_EQ(col[2], 2);
  EXPECT_EQ(col[3], 5);
  EXPECT_EQ(col[4], 3);
  EXPECT_EQ(col[5], 6);
}

TEST_F(UtilsTest, ToColMajor3x2) {
  std::vector<float> row = {1, 2, 3, 4, 5, 6};  // 3 rows, 2 cols
  auto col = cuda::to_col_major<float>(std::span<const float>(row), 3, 2);
  
  ASSERT_EQ(col.size(), 6);
  EXPECT_EQ(col[0], 1);
  EXPECT_EQ(col[1], 3);
  EXPECT_EQ(col[2], 5);
  EXPECT_EQ(col[3], 2);
  EXPECT_EQ(col[4], 4);
  EXPECT_EQ(col[5], 6);
}

TEST_F(UtilsTest, ComputeSquaredNorms) {
  std::vector<float> data = {3, 4, 0, 5, 12, 0};  // 2 vectors of dim 3
  auto norms = cuda::compute_squared_norms<float>(std::span<const float>(data), 2, 3);
  
  ASSERT_EQ(norms.size(), 2);
  EXPECT_FLOAT_EQ(norms[0], 25.0f);   // 3^2 + 4^2 + 0^2 = 25
  EXPECT_FLOAT_EQ(norms[1], 169.0f);  // 5^2 + 12^2 + 0^2 = 169
}

TEST_F(UtilsTest, ComputeSquaredNormsEmpty) {
  std::vector<float> data;
  auto norms = cuda::compute_squared_norms<float>(std::span<const float>(data), 0, 0);
  
  EXPECT_TRUE(norms.empty());
}

} // namespace nordlys::clustering::cuda::test

#endif // NORDLYS_HAS_CUDA
