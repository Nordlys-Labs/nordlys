#include <gtest/gtest.h>

#include <nordlys_core/cluster.hpp>

// Test fixture for CPU backend
template <typename Scalar> class ClusterEngineCpuTestT : public ::testing::Test {
protected:
  ClusterEngineT<Scalar> engine{ClusterBackendType::Cpu};
};

// Test both float and double
using ScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(ClusterEngineCpuTestT, ScalarTypes);

// =============================================================================
// CPU Backend Tests
// =============================================================================

TYPED_TEST(ClusterEngineCpuTestT, EmptyEngine) {
  EXPECT_EQ(this->engine.get_n_clusters(), 0);
  EXPECT_FALSE(this->engine.is_gpu_accelerated());
}

TYPED_TEST(ClusterEngineCpuTestT, LoadCentroids) {
  EmbeddingMatrixT<TypeParam> centers(3, 4);
  centers << TypeParam(1.0), TypeParam(0.0), TypeParam(0.0), TypeParam(0.0), TypeParam(0.0),
      TypeParam(1.0), TypeParam(0.0), TypeParam(0.0), TypeParam(0.0), TypeParam(0.0),
      TypeParam(1.0), TypeParam(0.0);

  this->engine.load_centroids(centers);
  EXPECT_EQ(this->engine.get_n_clusters(), 3);
}

TYPED_TEST(ClusterEngineCpuTestT, AssignToNearestCluster) {
  EmbeddingMatrixT<TypeParam> centers(3, 4);
  centers << TypeParam(1.0), TypeParam(0.0), TypeParam(0.0), TypeParam(0.0), TypeParam(0.0),
      TypeParam(1.0), TypeParam(0.0), TypeParam(0.0), TypeParam(0.0), TypeParam(0.0),
      TypeParam(1.0), TypeParam(0.0);

  this->engine.load_centroids(centers);

  EmbeddingVectorT<TypeParam> vec(4);
  vec << TypeParam(0.9), TypeParam(0.1), TypeParam(0.0), TypeParam(0.0);

  auto [cluster_id, distance] = this->engine.assign(vec);
  EXPECT_EQ(cluster_id, 0);
  EXPECT_GT(distance, TypeParam(0.0));
}

TYPED_TEST(ClusterEngineCpuTestT, AssignToSecondCluster) {
  EmbeddingMatrixT<TypeParam> centers(3, 4);
  centers << TypeParam(1.0), TypeParam(0.0), TypeParam(0.0), TypeParam(0.0), TypeParam(0.0),
      TypeParam(1.0), TypeParam(0.0), TypeParam(0.0), TypeParam(0.0), TypeParam(0.0),
      TypeParam(1.0), TypeParam(0.0);

  this->engine.load_centroids(centers);

  EmbeddingVectorT<TypeParam> vec(4);
  vec << TypeParam(0.0), TypeParam(0.95), TypeParam(0.05), TypeParam(0.0);

  auto [cluster_id, distance] = this->engine.assign(vec);
  EXPECT_EQ(cluster_id, 1);
  EXPECT_LT(distance, TypeParam(0.1));
}

TYPED_TEST(ClusterEngineCpuTestT, AssignToThirdCluster) {
  EmbeddingMatrixT<TypeParam> centers(3, 4);
  centers << TypeParam(1.0), TypeParam(0.0), TypeParam(0.0), TypeParam(0.0), TypeParam(0.0),
      TypeParam(1.0), TypeParam(0.0), TypeParam(0.0), TypeParam(0.0), TypeParam(0.0),
      TypeParam(1.0), TypeParam(0.0);

  this->engine.load_centroids(centers);

  EmbeddingVectorT<TypeParam> vec(4);
  vec << TypeParam(0.0), TypeParam(0.0), TypeParam(1.0), TypeParam(0.0);

  auto [cluster_id, distance] = this->engine.assign(vec);
  EXPECT_EQ(cluster_id, 2);
  EXPECT_NEAR(distance, TypeParam(0.0), TypeParam(1e-6));
}

TYPED_TEST(ClusterEngineCpuTestT, ManyClusterAssignment) {
  constexpr int N_CLUSTERS = 100;
  constexpr int DIM = 128;
  constexpr int N_QUERIES = 50;

  EmbeddingMatrixT<TypeParam> centers = EmbeddingMatrixT<TypeParam>::Random(N_CLUSTERS, DIM);

  this->engine.load_centroids(centers);
  EXPECT_EQ(this->engine.get_n_clusters(), N_CLUSTERS);

  for (int q = 0; q < N_QUERIES; ++q) {
    EmbeddingVectorT<TypeParam> query = EmbeddingVectorT<TypeParam>::Random(DIM);

    auto [cluster_id, distance] = this->engine.assign(query);

    EXPECT_GE(cluster_id, 0);
    EXPECT_LT(cluster_id, N_CLUSTERS);
    EXPECT_GE(distance, TypeParam(0.0));
  }
}
