#ifdef NORDLYS_HAS_CUDA

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <random>

#include <nordlys/clustering/cuda/distance.cuh>
#include <nordlys/clustering/cuda/memory.cuh>

namespace nordlys::clustering::cuda::test {

class DistanceTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(0));
  }

  static auto compute_cpu_l2_distance(
      const float* query, const float* centroid, int dim) -> float {
    float dist = 0.0f;
    for (int i = 0; i < dim; ++i) {
      float diff = query[i] - centroid[i];
      dist += diff * diff;
    }
    return std::sqrt(dist);
  }

  static auto find_nearest_cpu(
      const float* query, int dim,
      const std::vector<float>& centroids, int n_clusters) -> std::pair<int, float> {
    int best_idx = -1;
    float best_dist = std::numeric_limits<float>::max();
    for (int i = 0; i < n_clusters; ++i) {
      float dist = compute_cpu_l2_distance(query, centroids.data() + i * dim, dim);
      if (dist < best_dist) {
        best_dist = dist;
        best_idx = i;
      }
    }
    return {best_idx, best_dist};
  }

  static auto compute_squared_norms_cpu(const float* data, int n, int dim) -> std::vector<float> {
    std::vector<float> norms(n);
    for (int i = 0; i < n; ++i) {
      float sum = 0.0f;
      for (int j = 0; j < dim; ++j) {
        float val = data[i * dim + j];
        sum += val * val;
      }
      norms[i] = sum;
    }
    return norms;
  }

  static auto compute_dots_cpu(const float* query, const float* centroids, 
                               int n_clusters, int dim) -> std::vector<float> {
    std::vector<float> dots(n_clusters);
    for (int i = 0; i < n_clusters; ++i) {
      float sum = 0.0f;
      for (int j = 0; j < dim; ++j) {
        sum += query[j] * centroids[i * dim + j];
      }
      dots[i] = sum;
    }
    return dots;
  }
};

TEST_F(DistanceTest, FusedL2ArgminSharedMemSize) {
  constexpr size_t expected = 32 * (2 * sizeof(float) + sizeof(int));
  EXPECT_EQ(fused_l2_argmin_shared_mem_size<float>(), expected);
  EXPECT_EQ(fused_l2_argmin_shared_mem_size<double>(), 32 * (2 * sizeof(double) + sizeof(int)));
}

TEST_F(DistanceTest, SingleQuerySingleCluster) {
  constexpr int DIM = 64;
  constexpr int N_CLUSTERS = 1;

  std::vector<float> query(DIM, 1.0f);
  std::vector<float> centroids(N_CLUSTERS * DIM, 0.5f);

  auto expected = find_nearest_cpu(query.data(), DIM, centroids, N_CLUSTERS);
  auto norms = compute_squared_norms_cpu(centroids.data(), N_CLUSTERS, DIM);
  auto dots = compute_dots_cpu(query.data(), centroids.data(), N_CLUSTERS, DIM);

  cuda::DevicePtr<float> d_query(DIM);
  cuda::DevicePtr<float> d_norms(N_CLUSTERS);
  cuda::DevicePtr<float> d_dots(N_CLUSTERS);
  cuda::DevicePtr<int> d_idx(1);
  cuda::DevicePtr<float> d_dist(1);
  cuda::PinnedPtr<int> h_idx(1);
  cuda::PinnedPtr<float> h_dist(1);

  ASSERT_EQ(cudaSuccess, cudaMemcpy(d_query.get(), query.data(), DIM * sizeof(float), cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(d_norms.get(), norms.data(), N_CLUSTERS * sizeof(float), cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(d_dots.get(), dots.data(), N_CLUSTERS * sizeof(float), cudaMemcpyHostToDevice));

  const size_t shared_mem = fused_l2_argmin_shared_mem_size<float>();
  dim3 block(128);
  dim3 grid(1);
  fused_l2_argmin_with_norm<float><<<grid, block, shared_mem>>>(
      d_query.get(), d_norms.get(), d_dots.get(), N_CLUSTERS, DIM,
      d_idx.get(), d_dist.get());

  ASSERT_EQ(cudaSuccess, cudaGetLastError());
  ASSERT_EQ(cudaSuccess, cudaMemcpy(h_idx.get(), d_idx.get(), sizeof(int), cudaMemcpyDeviceToHost));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(h_dist.get(), d_dist.get(), sizeof(float), cudaMemcpyDeviceToHost));
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

  EXPECT_EQ(*h_idx.get(), expected.first);
  EXPECT_NEAR(*h_dist.get(), expected.second, 1e-3f);
}

TEST_F(DistanceTest, SingleQueryMultipleClusters) {
  constexpr int DIM = 64;
  constexpr int N_CLUSTERS = 10;

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> query(DIM);
  for (int i = 0; i < DIM; ++i) query[i] = dist(gen);

  std::vector<float> centroids(N_CLUSTERS * DIM);
  for (int i = 0; i < N_CLUSTERS * DIM; ++i) centroids[i] = dist(gen);

  auto expected = find_nearest_cpu(query.data(), DIM, centroids, N_CLUSTERS);
  auto norms = compute_squared_norms_cpu(centroids.data(), N_CLUSTERS, DIM);
  auto dots = compute_dots_cpu(query.data(), centroids.data(), N_CLUSTERS, DIM);

  cuda::DevicePtr<float> d_query(DIM);
  cuda::DevicePtr<float> d_norms(N_CLUSTERS);
  cuda::DevicePtr<float> d_dots(N_CLUSTERS);
  cuda::DevicePtr<int> d_idx(1);
  cuda::DevicePtr<float> d_dist(1);
  cuda::PinnedPtr<int> h_idx(1);
  cuda::PinnedPtr<float> h_dist(1);

  ASSERT_EQ(cudaSuccess, cudaMemcpy(d_query.get(), query.data(), DIM * sizeof(float), cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(d_norms.get(), norms.data(), N_CLUSTERS * sizeof(float), cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(d_dots.get(), dots.data(), N_CLUSTERS * sizeof(float), cudaMemcpyHostToDevice));

  const size_t shared_mem = fused_l2_argmin_shared_mem_size<float>();
  dim3 block(128);
  dim3 grid(1);
  fused_l2_argmin_with_norm<float><<<grid, block, shared_mem>>>(
      d_query.get(), d_norms.get(), d_dots.get(), N_CLUSTERS, DIM,
      d_idx.get(), d_dist.get());

  ASSERT_EQ(cudaSuccess, cudaGetLastError());
  ASSERT_EQ(cudaSuccess, cudaMemcpy(h_idx.get(), d_idx.get(), sizeof(int), cudaMemcpyDeviceToHost));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(h_dist.get(), d_dist.get(), sizeof(float), cudaMemcpyDeviceToHost));
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

  EXPECT_EQ(*h_idx.get(), expected.first);
  EXPECT_NEAR(*h_dist.get(), expected.second, 1e-3f);
}

TEST_F(DistanceTest, NonMultipleOfFourClusters) {
  constexpr int DIM = 64;
  constexpr int N_CLUSTERS = 37;

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> query(DIM);
  for (int i = 0; i < DIM; ++i) query[i] = dist(gen);

  std::vector<float> centroids(N_CLUSTERS * DIM);
  for (int i = 0; i < N_CLUSTERS * DIM; ++i) centroids[i] = dist(gen);

  auto expected = find_nearest_cpu(query.data(), DIM, centroids, N_CLUSTERS);
  auto norms = compute_squared_norms_cpu(centroids.data(), N_CLUSTERS, DIM);
  auto dots = compute_dots_cpu(query.data(), centroids.data(), N_CLUSTERS, DIM);

  cuda::DevicePtr<float> d_query(DIM);
  cuda::DevicePtr<float> d_norms(N_CLUSTERS);
  cuda::DevicePtr<float> d_dots(N_CLUSTERS);
  cuda::DevicePtr<int> d_idx(1);
  cuda::DevicePtr<float> d_dist(1);
  cuda::PinnedPtr<int> h_idx(1);
  cuda::PinnedPtr<float> h_dist(1);

  ASSERT_EQ(cudaSuccess, cudaMemcpy(d_query.get(), query.data(), DIM * sizeof(float), cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(d_norms.get(), norms.data(), N_CLUSTERS * sizeof(float), cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(d_dots.get(), dots.data(), N_CLUSTERS * sizeof(float), cudaMemcpyHostToDevice));

  const size_t shared_mem = fused_l2_argmin_shared_mem_size<float>();
  dim3 block(128);
  dim3 grid(1);
  fused_l2_argmin_with_norm<float><<<grid, block, shared_mem>>>(
      d_query.get(), d_norms.get(), d_dots.get(), N_CLUSTERS, DIM,
      d_idx.get(), d_dist.get());

  ASSERT_EQ(cudaSuccess, cudaGetLastError());
  ASSERT_EQ(cudaSuccess, cudaMemcpy(h_idx.get(), d_idx.get(), sizeof(int), cudaMemcpyDeviceToHost));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(h_dist.get(), d_dist.get(), sizeof(float), cudaMemcpyDeviceToHost));
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

  EXPECT_EQ(*h_idx.get(), expected.first);
  EXPECT_NEAR(*h_dist.get(), expected.second, 1e-3f);
}

TEST_F(DistanceTest, LargeDimension) {
  constexpr int DIM = 1024;
  constexpr int N_CLUSTERS = 5;

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> query(DIM);
  for (int i = 0; i < DIM; ++i) query[i] = dist(gen);

  std::vector<float> centroids(N_CLUSTERS * DIM);
  for (int i = 0; i < N_CLUSTERS * DIM; ++i) centroids[i] = dist(gen);

  auto expected = find_nearest_cpu(query.data(), DIM, centroids, N_CLUSTERS);
  auto norms = compute_squared_norms_cpu(centroids.data(), N_CLUSTERS, DIM);
  auto dots = compute_dots_cpu(query.data(), centroids.data(), N_CLUSTERS, DIM);

  cuda::DevicePtr<float> d_query(DIM);
  cuda::DevicePtr<float> d_norms(N_CLUSTERS);
  cuda::DevicePtr<float> d_dots(N_CLUSTERS);
  cuda::DevicePtr<int> d_idx(1);
  cuda::DevicePtr<float> d_dist(1);
  cuda::PinnedPtr<int> h_idx(1);
  cuda::PinnedPtr<float> h_dist(1);

  ASSERT_EQ(cudaSuccess, cudaMemcpy(d_query.get(), query.data(), DIM * sizeof(float), cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(d_norms.get(), norms.data(), N_CLUSTERS * sizeof(float), cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(d_dots.get(), dots.data(), N_CLUSTERS * sizeof(float), cudaMemcpyHostToDevice));

  const size_t shared_mem = fused_l2_argmin_shared_mem_size<float>();
  dim3 block(256);
  dim3 grid(1);
  fused_l2_argmin_with_norm<float><<<grid, block, shared_mem>>>(
      d_query.get(), d_norms.get(), d_dots.get(), N_CLUSTERS, DIM,
      d_idx.get(), d_dist.get());

  ASSERT_EQ(cudaSuccess, cudaGetLastError());
  ASSERT_EQ(cudaSuccess, cudaMemcpy(h_idx.get(), d_idx.get(), sizeof(int), cudaMemcpyDeviceToHost));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(h_dist.get(), d_dist.get(), sizeof(float), cudaMemcpyDeviceToHost));
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

  EXPECT_EQ(*h_idx.get(), expected.first);
  EXPECT_NEAR(*h_dist.get(), expected.second, 1e-2f);
}

} // namespace nordlys::clustering::cuda::test

#endif // NORDLYS_HAS_CUDA
