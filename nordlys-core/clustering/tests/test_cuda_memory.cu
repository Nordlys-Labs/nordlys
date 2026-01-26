#ifdef NORDLYS_HAS_CUDA

#  include <gtest/gtest.h>

#  include <nordlys/clustering/cuda/memory.cuh>
#  include <vector>

class CudaMemoryTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(CudaMemoryTest, DevicePtrDefaultConstruction) {
  CudaDevicePtr<float> ptr;
  EXPECT_TRUE(ptr.empty());
  EXPECT_EQ(ptr.size(), 0);
  EXPECT_EQ(ptr.get(), nullptr);
}

TEST_F(CudaMemoryTest, DevicePtrAllocation) {
  CudaDevicePtr<float> ptr(100);
  EXPECT_FALSE(ptr.empty());
  EXPECT_EQ(ptr.size(), 100);
  EXPECT_NE(ptr.get(), nullptr);
}

TEST_F(CudaMemoryTest, DevicePtrMoveConstruction) {
  CudaDevicePtr<float> ptr1(100);
  float* original_addr = ptr1.get();

  CudaDevicePtr<float> ptr2 = std::move(ptr1);

  EXPECT_TRUE(ptr1.empty());
  EXPECT_EQ(ptr1.get(), nullptr);
  EXPECT_FALSE(ptr2.empty());
  EXPECT_EQ(ptr2.size(), 100);
  EXPECT_EQ(ptr2.get(), original_addr);
}

TEST_F(CudaMemoryTest, DevicePtrMoveAssignment) {
  CudaDevicePtr<float> ptr1(100);
  CudaDevicePtr<float> ptr2(50);
  float* original_addr = ptr1.get();

  ptr2 = std::move(ptr1);

  EXPECT_TRUE(ptr1.empty());
  EXPECT_FALSE(ptr2.empty());
  EXPECT_EQ(ptr2.size(), 100);
  EXPECT_EQ(ptr2.get(), original_addr);
}

TEST_F(CudaMemoryTest, DevicePtrReset) {
  CudaDevicePtr<float> ptr(100);
  EXPECT_EQ(ptr.size(), 100);

  ptr.reset(200);
  EXPECT_EQ(ptr.size(), 200);
  EXPECT_NE(ptr.get(), nullptr);

  ptr.reset();
  EXPECT_TRUE(ptr.empty());
  EXPECT_EQ(ptr.get(), nullptr);
}

TEST_F(CudaMemoryTest, DevicePtrDataTransfer) {
  const size_t count = 100;
  std::vector<float> h_data(count);
  for (size_t i = 0; i < count; ++i) {
    h_data[i] = static_cast<float>(i);
  }

  CudaDevicePtr<float> d_data(count);
  NORDLYS_CUDA_CHECK(
      cudaMemcpy(d_data.get(), h_data.data(), count * sizeof(float), cudaMemcpyHostToDevice));

  std::vector<float> h_result(count);
  NORDLYS_CUDA_CHECK(
      cudaMemcpy(h_result.data(), d_data.get(), count * sizeof(float), cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < count; ++i) {
    EXPECT_FLOAT_EQ(h_result[i], h_data[i]);
  }
}

TEST_F(CudaMemoryTest, PinnedPtrDefaultConstruction) {
  CudaPinnedPtr<float> ptr;
  EXPECT_TRUE(ptr.empty());
  EXPECT_EQ(ptr.size(), 0);
  EXPECT_EQ(ptr.get(), nullptr);
}

TEST_F(CudaMemoryTest, PinnedPtrAllocation) {
  CudaPinnedPtr<float> ptr(100);
  EXPECT_FALSE(ptr.empty());
  EXPECT_EQ(ptr.size(), 100);
  EXPECT_NE(ptr.get(), nullptr);
}

TEST_F(CudaMemoryTest, PinnedPtrMoveConstruction) {
  CudaPinnedPtr<float> ptr1(100);
  float* original_addr = ptr1.get();

  CudaPinnedPtr<float> ptr2 = std::move(ptr1);

  EXPECT_TRUE(ptr1.empty());
  EXPECT_FALSE(ptr2.empty());
  EXPECT_EQ(ptr2.size(), 100);
  EXPECT_EQ(ptr2.get(), original_addr);
}

TEST_F(CudaMemoryTest, PinnedPtrMoveAssignment) {
  CudaPinnedPtr<float> ptr1(100);
  CudaPinnedPtr<float> ptr2(50);
  float* original_addr = ptr1.get();

  ptr2 = std::move(ptr1);

  EXPECT_TRUE(ptr1.empty());
  EXPECT_FALSE(ptr2.empty());
  EXPECT_EQ(ptr2.size(), 100);
  EXPECT_EQ(ptr2.get(), original_addr);
}

TEST_F(CudaMemoryTest, PinnedPtrReset) {
  CudaPinnedPtr<float> ptr(100);
  EXPECT_EQ(ptr.size(), 100);

  ptr.reset(200);
  EXPECT_EQ(ptr.size(), 200);
  EXPECT_NE(ptr.get(), nullptr);

  ptr.reset();
  EXPECT_TRUE(ptr.empty());
}

TEST_F(CudaMemoryTest, PinnedPtrReadWrite) {
  const size_t count = 100;
  CudaPinnedPtr<float> ptr(count);

  for (size_t i = 0; i < count; ++i) {
    ptr.get()[i] = static_cast<float>(i * 2);
  }

  for (size_t i = 0; i < count; ++i) {
    EXPECT_FLOAT_EQ(ptr.get()[i], static_cast<float>(i * 2));
  }
}

TEST_F(CudaMemoryTest, DevicePtrDoubleType) {
  CudaDevicePtr<double> ptr(50);
  EXPECT_FALSE(ptr.empty());
  EXPECT_EQ(ptr.size(), 50);

  std::vector<double> h_data(50);
  for (size_t i = 0; i < 50; ++i) {
    h_data[i] = static_cast<double>(i) * 0.5;
  }

  NORDLYS_CUDA_CHECK(
      cudaMemcpy(ptr.get(), h_data.data(), 50 * sizeof(double), cudaMemcpyHostToDevice));

  std::vector<double> h_result(50);
  NORDLYS_CUDA_CHECK(
      cudaMemcpy(h_result.data(), ptr.get(), 50 * sizeof(double), cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < 50; ++i) {
    EXPECT_DOUBLE_EQ(h_result[i], h_data[i]);
  }
}

TEST_F(CudaMemoryTest, PinnedPtrDoubleType) {
  CudaPinnedPtr<double> ptr(50);
  EXPECT_FALSE(ptr.empty());
  EXPECT_EQ(ptr.size(), 50);

  for (size_t i = 0; i < 50; ++i) {
    ptr.get()[i] = static_cast<double>(i) * 0.5;
  }

  for (size_t i = 0; i < 50; ++i) {
    EXPECT_DOUBLE_EQ(ptr.get()[i], static_cast<double>(i) * 0.5);
  }
}

#endif  // NORDLYS_HAS_CUDA
