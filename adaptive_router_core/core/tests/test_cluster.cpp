#include <gtest/gtest.h>

#include <adaptive_core/cluster.hpp>

// Test fixture for templated cluster engine tests
template<typename Scalar>
class ClusterEngineTestT : public ::testing::Test {
protected:
  ClusterEngineT<Scalar> engine;
};

// Test both float and double
using ScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(ClusterEngineTestT, ScalarTypes);

TYPED_TEST(ClusterEngineTestT, EmptyEngine) {
  EXPECT_EQ(this->engine.get_n_clusters(), 0);
}

TYPED_TEST(ClusterEngineTestT, LoadCentroids) {
  // Create simple 3 clusters with 4D embeddings
  EmbeddingMatrixT<TypeParam> centers(3, 4);
  centers << TypeParam(1.0), TypeParam(0.0), TypeParam(0.0), TypeParam(0.0),
             TypeParam(0.0), TypeParam(1.0), TypeParam(0.0), TypeParam(0.0),
             TypeParam(0.0), TypeParam(0.0), TypeParam(1.0), TypeParam(0.0);

  this->engine.load_centroids(centers);
  EXPECT_EQ(this->engine.get_n_clusters(), 3);
}

TYPED_TEST(ClusterEngineTestT, AssignToNearestCluster) {
  EmbeddingMatrixT<TypeParam> centers(3, 4);
  centers << TypeParam(1.0), TypeParam(0.0), TypeParam(0.0), TypeParam(0.0),
             TypeParam(0.0), TypeParam(1.0), TypeParam(0.0), TypeParam(0.0),
             TypeParam(0.0), TypeParam(0.0), TypeParam(1.0), TypeParam(0.0);

  this->engine.load_centroids(centers);

  // Test vector close to first centroid
  EmbeddingVectorT<TypeParam> vec(4);
  vec << TypeParam(0.9), TypeParam(0.1), TypeParam(0.0), TypeParam(0.0);

  auto [cluster_id, distance] = this->engine.assign(vec);
  EXPECT_EQ(cluster_id, 0);
  EXPECT_GT(distance, TypeParam(0.0));
}

TYPED_TEST(ClusterEngineTestT, AssignToSecondCluster) {
  EmbeddingMatrixT<TypeParam> centers(3, 4);
  centers << TypeParam(1.0), TypeParam(0.0), TypeParam(0.0), TypeParam(0.0),
             TypeParam(0.0), TypeParam(1.0), TypeParam(0.0), TypeParam(0.0),
             TypeParam(0.0), TypeParam(0.0), TypeParam(1.0), TypeParam(0.0);

  this->engine.load_centroids(centers);

  // Test vector close to second centroid
  EmbeddingVectorT<TypeParam> vec(4);
  vec << TypeParam(0.0), TypeParam(0.95), TypeParam(0.05), TypeParam(0.0);

  auto [cluster_id, distance] = this->engine.assign(vec);
  EXPECT_EQ(cluster_id, 1);
  EXPECT_LT(distance, TypeParam(0.1));
}

TYPED_TEST(ClusterEngineTestT, AssignToThirdCluster) {
  EmbeddingMatrixT<TypeParam> centers(3, 4);
  centers << TypeParam(1.0), TypeParam(0.0), TypeParam(0.0), TypeParam(0.0),
             TypeParam(0.0), TypeParam(1.0), TypeParam(0.0), TypeParam(0.0),
             TypeParam(0.0), TypeParam(0.0), TypeParam(1.0), TypeParam(0.0);

  this->engine.load_centroids(centers);

  // Test vector close to third centroid
  EmbeddingVectorT<TypeParam> vec(4);
  vec << TypeParam(0.0), TypeParam(0.0), TypeParam(1.0), TypeParam(0.0);

  auto [cluster_id, distance] = this->engine.assign(vec);
  EXPECT_EQ(cluster_id, 2);
  EXPECT_NEAR(distance, TypeParam(0.0), TypeParam(1e-6));
}
