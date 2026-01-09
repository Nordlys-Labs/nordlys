#include <gtest/gtest.h>

#include <nordlys_core/cluster.hpp>
#include <random>

template <typename Scalar> class ClusterEngineCpuTestT : public ::testing::Test {
protected:
  ClusterEngineT<Scalar> engine{ClusterBackendType::Cpu};

  static void fill_matrix(EmbeddingMatrixT<Scalar>& m, std::initializer_list<Scalar> values) {
    auto it = values.begin();
    for (size_t i = 0; i < m.rows(); ++i) {
      for (size_t j = 0; j < m.cols(); ++j) {
        m(i, j) = (it != values.end()) ? *it++ : Scalar(0);
      }
    }
  }

  static void random_matrix(EmbeddingMatrixT<Scalar>& m, std::mt19937& gen) {
    std::uniform_real_distribution<Scalar> dist(-1.0, 1.0);
    for (size_t i = 0; i < m.rows(); ++i) {
      for (size_t j = 0; j < m.cols(); ++j) {
        m(i, j) = dist(gen);
      }
    }
  }
};

using ScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(ClusterEngineCpuTestT, ScalarTypes);

TYPED_TEST(ClusterEngineCpuTestT, EmptyEngine) {
  EXPECT_EQ(this->engine.get_n_clusters(), 0);
  EXPECT_FALSE(this->engine.is_gpu_accelerated());
}

TYPED_TEST(ClusterEngineCpuTestT, LoadCentroids) {
  EmbeddingMatrixT<TypeParam> centers(3, 4);
  this->fill_matrix(centers, {TypeParam(1.0), TypeParam(0.0), TypeParam(0.0), TypeParam(0.0),
                              TypeParam(0.0), TypeParam(1.0), TypeParam(0.0), TypeParam(0.0),
                              TypeParam(0.0), TypeParam(0.0), TypeParam(1.0), TypeParam(0.0)});

  this->engine.load_centroids(centers);
  EXPECT_EQ(this->engine.get_n_clusters(), 3);
}

TYPED_TEST(ClusterEngineCpuTestT, AssignToNearestCluster) {
  EmbeddingMatrixT<TypeParam> centers(3, 4);
  this->fill_matrix(centers, {TypeParam(1.0), TypeParam(0.0), TypeParam(0.0), TypeParam(0.0),
                              TypeParam(0.0), TypeParam(1.0), TypeParam(0.0), TypeParam(0.0),
                              TypeParam(0.0), TypeParam(0.0), TypeParam(1.0), TypeParam(0.0)});

  this->engine.load_centroids(centers);

  TypeParam vec[] = {TypeParam(0.9), TypeParam(0.1), TypeParam(0.0), TypeParam(0.0)};
  auto [cluster_id, distance] = this->engine.assign(vec, 4);
  EXPECT_EQ(cluster_id, 0);
  EXPECT_GT(distance, TypeParam(0.0));
}

TYPED_TEST(ClusterEngineCpuTestT, AssignToSecondCluster) {
  EmbeddingMatrixT<TypeParam> centers(3, 4);
  this->fill_matrix(centers, {TypeParam(1.0), TypeParam(0.0), TypeParam(0.0), TypeParam(0.0),
                              TypeParam(0.0), TypeParam(1.0), TypeParam(0.0), TypeParam(0.0),
                              TypeParam(0.0), TypeParam(0.0), TypeParam(1.0), TypeParam(0.0)});

  this->engine.load_centroids(centers);

  TypeParam vec[] = {TypeParam(0.0), TypeParam(0.95), TypeParam(0.05), TypeParam(0.0)};
  auto [cluster_id, distance] = this->engine.assign(vec, 4);
  EXPECT_EQ(cluster_id, 1);
  EXPECT_LT(distance, TypeParam(0.1));
}

TYPED_TEST(ClusterEngineCpuTestT, AssignToThirdCluster) {
  EmbeddingMatrixT<TypeParam> centers(3, 4);
  this->fill_matrix(centers, {TypeParam(1.0), TypeParam(0.0), TypeParam(0.0), TypeParam(0.0),
                              TypeParam(0.0), TypeParam(1.0), TypeParam(0.0), TypeParam(0.0),
                              TypeParam(0.0), TypeParam(0.0), TypeParam(1.0), TypeParam(0.0)});

  this->engine.load_centroids(centers);

  TypeParam vec[] = {TypeParam(0.0), TypeParam(0.0), TypeParam(1.0), TypeParam(0.0)};
  auto [cluster_id, distance] = this->engine.assign(vec, 4);
  EXPECT_EQ(cluster_id, 2);
  EXPECT_NEAR(distance, TypeParam(0.0), TypeParam(1e-6));
}

TYPED_TEST(ClusterEngineCpuTestT, ManyClusterAssignment) {
  constexpr size_t N_CLUSTERS = 100;
  constexpr size_t DIM = 128;
  constexpr int N_QUERIES = 50;

  std::mt19937 gen(42);
  EmbeddingMatrixT<TypeParam> centers(N_CLUSTERS, DIM);
  this->random_matrix(centers, gen);

  this->engine.load_centroids(centers);
  EXPECT_EQ(this->engine.get_n_clusters(), static_cast<int>(N_CLUSTERS));

  std::uniform_real_distribution<TypeParam> dist(-1.0, 1.0);
  for (int q = 0; q < N_QUERIES; ++q) {
    std::vector<TypeParam> query(DIM);
    for (size_t d = 0; d < DIM; ++d) {
      query[d] = dist(gen);
    }

    auto [cluster_id, distance] = this->engine.assign(query.data(), DIM);

    EXPECT_GE(cluster_id, 0);
    EXPECT_LT(cluster_id, static_cast<int>(N_CLUSTERS));
    EXPECT_GE(distance, TypeParam(0.0));
  }
}
