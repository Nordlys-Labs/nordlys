#ifdef NORDLYS_HAS_CUDA

#include <gtest/gtest.h>
#include <vector>

#include <nordlys/clustering/cuda/memory.cuh>

namespace nordlys::clustering::cuda::test {

class MemoryTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(0));
  }
};

TEST_F(MemoryTest, DevicePtrDefaultConstruction) {
  cuda::DevicePtr<float> ptr;
  EXPECT_TRUE(ptr.empty());
  EXPECT_EQ(ptr.size(), 0);
  EXPECT_EQ(ptr.get(), nullptr);
}

TEST_F(MemoryTest, DevicePtrAllocation) {
  cuda::DevicePtr<float> ptr(100);
  EXPECT_FALSE(ptr.empty());
  EXPECT_EQ(ptr.size(), 100);
  EXPECT_NE(ptr.get(), nullptr);
}

TEST_F(MemoryTest, DevicePtrMoveConstruction) {
  cuda::DevicePtr<float> ptr1(100);
  float* original_addr = ptr1.get();
  
  cuda::DevicePtr<float> ptr2 = std::move(ptr1);
  
  EXPECT_TRUE(ptr1.empty());
  EXPECT_EQ(ptr1.get(), nullptr);
  EXPECT_FALSE(ptr2.empty());
  EXPECT_EQ(ptr2.size(), 100);
  EXPECT_EQ(ptr2.get(), original_addr);
}

TEST_F(MemoryTest, DevicePtrReset) {
  cuda::DevicePtr<float> ptr(100);
  ptr.reset(200);
  
  EXPECT_FALSE(ptr.empty());
  EXPECT_EQ(ptr.size(), 200);
}

TEST_F(MemoryTest, DevicePtrDataTransfer) {
  cuda::DevicePtr<float> d_ptr(50);
  std::vector<float> h_data(50);
  for (size_t i = 0; i < 50; ++i) {
    h_data[i] = static_cast<float>(i);
  }
  
  ASSERT_EQ(cudaSuccess, cudaMemcpy(d_ptr.get(), h_data.data(), 50 * sizeof(float), cudaMemcpyHostToDevice));
  
  std::vector<float> h_result(50);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(h_result.data(), d_ptr.get(), 50 * sizeof(float), cudaMemcpyDeviceToHost));
  
  for (size_t i = 0; i < 50; ++i) {
    EXPECT_FLOAT_EQ(h_result[i], h_data[i]);
  }
}

TEST_F(MemoryTest, PinnedPtrDefaultConstruction) {
  cuda::PinnedPtr<float> ptr;
  EXPECT_TRUE(ptr.empty());
}

TEST_F(MemoryTest, PinnedPtrAllocation) {
  cuda::PinnedPtr<float> ptr(100);
  EXPECT_FALSE(ptr.empty());
  EXPECT_EQ(ptr.size(), 100);
}

TEST_F(MemoryTest, PinnedPtrReadWrite) {
  cuda::PinnedPtr<float> ptr(50);
  for (size_t i = 0; i < 50; ++i) {
    ptr.get()[i] = static_cast<float>(i) * 0.5f;
  }
  
  for (size_t i = 0; i < 50; ++i) {
    EXPECT_FLOAT_EQ(ptr.get()[i], static_cast<float>(i) * 0.5f);
  }
}

} // namespace nordlys::clustering::cuda::test

#endif // NORDLYS_HAS_CUDA
