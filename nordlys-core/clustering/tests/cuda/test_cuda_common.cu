#ifdef NORDLYS_HAS_CUDA

#include <gtest/gtest.h>
#include <limits>

#include <nordlys/clustering/cuda/common.cuh>

namespace nordlys::clustering::cuda::test {

class CommonTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(0));
  }
};

TEST_F(CommonTest, TypeTraitsFloat) {
  EXPECT_EQ(type_traits<float>.max_value, std::numeric_limits<float>::max());
  EXPECT_EQ(type_traits<float>.min_value, std::numeric_limits<float>::lowest());
  EXPECT_EQ(type_traits<float>.epsilon, std::numeric_limits<float>::epsilon());
}

TEST_F(CommonTest, TypeTraitsDouble) {
  EXPECT_EQ(type_traits<double>.max_value, std::numeric_limits<double>::max());
  EXPECT_EQ(type_traits<double>.min_value, std::numeric_limits<double>::lowest());
  EXPECT_EQ(type_traits<double>.epsilon, std::numeric_limits<double>::epsilon());
}

TEST_F(CommonTest, CudaCheckSuccess) {
  EXPECT_NO_THROW(NORDLYS_CUDA_CHECK(cudaSuccess));
}

TEST_F(CommonTest, CudaCheckError) {
  EXPECT_THROW(NORDLYS_CUDA_CHECK(cudaErrorMemoryAllocation), std::runtime_error);
}

TEST_F(CommonTest, CublasCheckSuccess) {
  EXPECT_NO_THROW(NORDLYS_CUBLAS_CHECK(CUBLAS_STATUS_SUCCESS));
}

TEST_F(CommonTest, CublasCheckError) {
  EXPECT_THROW(NORDLYS_CUBLAS_CHECK(CUBLAS_STATUS_NOT_INITIALIZED), std::runtime_error);
}

} // namespace nordlys::clustering::cuda::test

#endif // NORDLYS_HAS_CUDA
