#ifdef NORDLYS_HAS_CUDA

#include <gtest/gtest.h>
#include <vector>

#include <nordlys/clustering/cuda/memory.cuh>
#include <nordlys/clustering/cuda/reduce.cuh>

namespace nordlys::clustering::cuda::test {

__global__ void test_warp_reduce_sum(float* output, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) return;
  
  float val = static_cast<float>(tid + 1);
  float reduced = cuda::warp_reduce_sum(val);
  
  if (tid % 32 == 0) {
    output[tid / 32] = reduced;
  }
}

__global__ void test_warp_reduce_min_idx(float* output_min, float* output_idx, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) return;
  
  float val = static_cast<float>(n - tid);
  int idx = tid;
  cuda::warp_reduce_min_idx(val, idx);
  
  if (tid % 32 == 0) {
    output_min[tid / 32] = val;
    output_idx[tid / 32] = static_cast<float>(idx);
  }
}

__global__ void test_compute_partial_squared_norm(float* output, const float* input, int dim) {
  int tid = threadIdx.x;
  if (tid >= 1) return;
  
  float norm = cuda::compute_partial_squared_norm(input, dim);
  output[0] = norm;
}

class ReduceTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(0));
  }
};

TEST_F(ReduceTest, WarpReduceSum) {
  std::vector<float> output(32);
  cuda::DevicePtr<float> d_output(32);
  
  test_warp_reduce_sum<<<1, 1024>>>(d_output.get(), 1024);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(output.data(), d_output.get(), 32 * sizeof(float), cudaMemcpyDeviceToHost));
  
  // Sum of 1..32 = 528
  EXPECT_FLOAT_EQ(output[0], 528.0f);
}

TEST_F(ReduceTest, WarpReduceMinIdx) {
  std::vector<float> output_min(32), output_idx(32);
  cuda::DevicePtr<float> d_min(32);
  cuda::DevicePtr<float> d_idx(32);
  
  test_warp_reduce_min_idx<<<1, 1024>>>(d_min.get(), d_idx.get(), 1024);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(output_min.data(), d_min.get(), 32 * sizeof(float), cudaMemcpyDeviceToHost));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(output_idx.data(), d_idx.get(), 32 * sizeof(float), cudaMemcpyDeviceToHost));
  
  // Warp 0 (threads 0-31): values (1024, 1023, ..., 993), min = 993 at idx 31
  EXPECT_FLOAT_EQ(output_min[0], 993.0f);
  EXPECT_FLOAT_EQ(output_idx[0], 31.0f);
  // Last warp (threads 992-1023): values (32, 31, ..., 1), min = 1 at idx 1023
  EXPECT_FLOAT_EQ(output_min[31], 1.0f);
  EXPECT_FLOAT_EQ(output_idx[31], 1023.0f);
}

TEST_F(ReduceTest, ComputePartialSquaredNorm) {
  std::vector<float> input = {3, 4, 0, 0};  // ||v||^2 = 25
  std::vector<float> output(1);
  cuda::DevicePtr<float> d_input(4);
  cuda::DevicePtr<float> d_output(1);
  
  ASSERT_EQ(cudaSuccess, cudaMemcpy(d_input.get(), input.data(), 4 * sizeof(float), cudaMemcpyHostToDevice));
  
  test_compute_partial_squared_norm<<<1, 1>>>(d_output.get(), d_input.get(), 4);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(output.data(), d_output.get(), sizeof(float), cudaMemcpyDeviceToHost));
  
  EXPECT_FLOAT_EQ(output[0], 25.0f);
}

} // namespace nordlys::clustering::cuda::test

#endif // NORDLYS_HAS_CUDA
