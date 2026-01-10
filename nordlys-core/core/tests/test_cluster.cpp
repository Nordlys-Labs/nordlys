#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <limits>
#include <nordlys_core/cluster.hpp>
#include <random>
#include <thread>
#include <vector>

// =============================================================================
// SECTION 1: CPU Backend - Basic Functionality  
// =============================================================================

template <typename Scalar> class ClusterEngineCpuTestT : public ::testing::Test {
protected:
  ClusterEngine<Scalar> engine{ClusterBackendType::Cpu};

  static void fill_matrix(EmbeddingMatrix<Scalar>& m, std::initializer_list<Scalar> values) {
    auto it = values.begin();
    for (size_t i = 0; i < m.rows(); ++i) {
      for (size_t j = 0; j < m.cols(); ++j) {
        m(i, j) = (it != values.end()) ? *it++ : Scalar(0);
      }
    }
  }

  static void random_matrix(EmbeddingMatrix<Scalar>& m, std::mt19937& gen) {
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
  EmbeddingMatrix<TypeParam> centers(3, 4);
  this->fill_matrix(centers, {TypeParam(1.0), TypeParam(0.0), TypeParam(0.0), TypeParam(0.0),
                              TypeParam(0.0), TypeParam(1.0), TypeParam(0.0), TypeParam(0.0),
                              TypeParam(0.0), TypeParam(0.0), TypeParam(1.0), TypeParam(0.0)});

  this->engine.load_centroids(centers);
  EXPECT_EQ(this->engine.get_n_clusters(), 3);
}

TYPED_TEST(ClusterEngineCpuTestT, AssignToNearestCluster) {
  EmbeddingMatrix<TypeParam> centers(3, 4);
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
  EmbeddingMatrix<TypeParam> centers(3, 4);
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
  EmbeddingMatrix<TypeParam> centers(3, 4);
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
  EmbeddingMatrix<TypeParam> centers(N_CLUSTERS, DIM);
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

TYPED_TEST(ClusterEngineCpuTestT, AssignBeforeLoadReturnsError) {
  std::vector<TypeParam> query(4, TypeParam(1.0));
  auto [cluster_id, distance] = this->engine.assign(query.data(), 4);
  EXPECT_EQ(cluster_id, -1);
  EXPECT_EQ(distance, TypeParam(0.0));
}



TYPED_TEST(ClusterEngineCpuTestT, LargeClusterCount) {
  constexpr size_t N_CLUSTERS = 1000;
  constexpr size_t DIM = 64;

  std::mt19937 gen(42);
  EmbeddingMatrix<TypeParam> centers(N_CLUSTERS, DIM);
  this->random_matrix(centers, gen);

  this->engine.load_centroids(centers);
  EXPECT_EQ(this->engine.get_n_clusters(), static_cast<int>(N_CLUSTERS));

  std::vector<TypeParam> query(DIM, TypeParam(0.5));
  auto [cluster_id, distance] = this->engine.assign(query.data(), DIM);
  EXPECT_GE(cluster_id, 0);
  EXPECT_LT(cluster_id, static_cast<int>(N_CLUSTERS));
}

TYPED_TEST(ClusterEngineCpuTestT, HighDimensionalEmbeddings) {
  constexpr size_t N_CLUSTERS = 10;
  constexpr size_t DIM = 2048;

  std::mt19937 gen(42);
  EmbeddingMatrix<TypeParam> centers(N_CLUSTERS, DIM);
  this->random_matrix(centers, gen);

  this->engine.load_centroids(centers);

  std::vector<TypeParam> query(DIM);
  std::uniform_real_distribution<TypeParam> dist(-1.0, 1.0);
  for (size_t d = 0; d < DIM; ++d) {
    query[d] = dist(gen);
  }

  auto [cluster_id, distance] = this->engine.assign(query.data(), DIM);
  EXPECT_GE(cluster_id, 0);
  EXPECT_LT(cluster_id, static_cast<int>(N_CLUSTERS));
}

// =============================================================================
// SECTION 2: CPU Backend - Thread Safety
// =============================================================================

template <typename Scalar> class ClusterCpuThreadSafetyTestT : public ::testing::Test {
protected:
  static constexpr size_t N_THREADS = 16;
  static constexpr int N_QUERIES_PER_THREAD = 1000;
  static constexpr size_t N_CLUSTERS = 100;
  static constexpr size_t DIM = 128;

  void SetUp() override {
    std::mt19937 gen(42);
    std::uniform_real_distribution<Scalar> dist(-1.0, 1.0);

    EmbeddingMatrix<Scalar> centers(N_CLUSTERS, DIM);
    for (size_t i = 0; i < N_CLUSTERS; ++i) {
      for (size_t j = 0; j < DIM; ++j) {
        centers(i, j) = dist(gen);
      }
    }

    engine_.load_centroids(centers);
  }

  ClusterEngine<Scalar> engine_{ClusterBackendType::Cpu};
};

TYPED_TEST_SUITE(ClusterCpuThreadSafetyTestT, ScalarTypes);

TYPED_TEST(ClusterCpuThreadSafetyTestT, ConcurrentAssign) {
  std::atomic<int> error_count{0};
  std::atomic<int> success_count{0};
  std::vector<std::thread> threads;

  for (size_t t = 0; t < this->N_THREADS; ++t) {
    threads.emplace_back([&, t]() {
      std::mt19937 gen(static_cast<unsigned>(42 + t));
      std::uniform_real_distribution<TypeParam> dist(-1.0, 1.0);

      for (int q = 0; q < this->N_QUERIES_PER_THREAD; ++q) {
        std::vector<TypeParam> query(this->DIM);
        for (size_t d = 0; d < this->DIM; ++d) {
          query[d] = dist(gen);
        }

        auto [cluster_id, distance] = this->engine_.assign(query.data(), this->DIM);

        if (cluster_id >= 0 && cluster_id < static_cast<int>(this->N_CLUSTERS) && distance >= 0
            && !std::isnan(distance) && !std::isinf(distance)) {
          success_count.fetch_add(1, std::memory_order_relaxed);
        } else {
          error_count.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(error_count.load(), 0) << "Thread safety violations detected in CPU backend";
  EXPECT_EQ(success_count.load(), static_cast<int>(this->N_THREADS) * this->N_QUERIES_PER_THREAD);
}

TYPED_TEST(ClusterCpuThreadSafetyTestT, StressTest) {
  static constexpr size_t STRESS_THREADS = 64;
  static constexpr int STRESS_QUERIES = 5000;

  std::atomic<int> success_count{0};
  std::atomic<int> failure_count{0};
  std::vector<std::thread> threads;

  for (size_t t = 0; t < STRESS_THREADS; ++t) {
    threads.emplace_back([&, t]() {
      std::mt19937 gen(static_cast<unsigned>(100 + t));
      std::uniform_real_distribution<TypeParam> dist(-1.0, 1.0);

      for (int q = 0; q < STRESS_QUERIES; ++q) {
        std::vector<TypeParam> query(this->DIM);
        for (size_t d = 0; d < this->DIM; ++d) {
          query[d] = dist(gen);
        }

        auto [cluster_id, distance] = this->engine_.assign(query.data(), this->DIM);

        if (cluster_id >= 0 && cluster_id < static_cast<int>(this->N_CLUSTERS) && distance >= 0
            && !std::isnan(distance)) {
          success_count.fetch_add(1, std::memory_order_relaxed);
        } else {
          failure_count.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(failure_count.load(), 0);
  EXPECT_EQ(success_count.load(), static_cast<int>(STRESS_THREADS) * STRESS_QUERIES);
}

// =============================================================================
// SECTION 2.5: CPU Backend - Batch Operations
// =============================================================================

template <typename Scalar> class ClusterBatchTestT : public ::testing::Test {
protected:
  static constexpr size_t N_CLUSTERS = 10;
  static constexpr size_t DIM = 64;

  void SetUp() override {
    std::mt19937 gen(42);
    std::uniform_real_distribution<Scalar> dist(-1.0, 1.0);

    EmbeddingMatrix<Scalar> centers(N_CLUSTERS, DIM);
    for (size_t i = 0; i < N_CLUSTERS; ++i) {
      for (size_t j = 0; j < DIM; ++j) {
        centers(i, j) = dist(gen);
      }
    }

    engine.load_centroids(centers);
  }

  ClusterEngine<Scalar> engine{ClusterBackendType::Cpu};
};

TYPED_TEST_SUITE(ClusterBatchTestT, ScalarTypes);

TYPED_TEST(ClusterBatchTestT, AssignBatchBasic) {
  constexpr size_t BATCH_SIZE = 100;

  std::mt19937 gen(123);
  std::uniform_real_distribution<TypeParam> dist(-1.0, 1.0);

  std::vector<TypeParam> embeddings(BATCH_SIZE * this->DIM);
  for (size_t i = 0; i < embeddings.size(); ++i) {
    embeddings[i] = dist(gen);
  }

  auto results = this->engine.assign_batch(embeddings.data(), BATCH_SIZE, this->DIM);

  ASSERT_EQ(results.size(), BATCH_SIZE);
  for (const auto& [cluster_id, distance] : results) {
    EXPECT_GE(cluster_id, 0);
    EXPECT_LT(cluster_id, static_cast<int>(this->N_CLUSTERS));
    EXPECT_GE(distance, TypeParam(0));
  }
}

TYPED_TEST(ClusterBatchTestT, AssignBatchMatchesSingleAssign) {
  constexpr size_t BATCH_SIZE = 50;

  std::mt19937 gen(456);
  std::uniform_real_distribution<TypeParam> dist(-1.0, 1.0);

  std::vector<TypeParam> embeddings(BATCH_SIZE * this->DIM);
  for (size_t i = 0; i < embeddings.size(); ++i) {
    embeddings[i] = dist(gen);
  }

  auto batch_results = this->engine.assign_batch(embeddings.data(), BATCH_SIZE, this->DIM);

  ASSERT_EQ(batch_results.size(), BATCH_SIZE);
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    auto single_result = this->engine.assign(embeddings.data() + i * this->DIM, this->DIM);
    EXPECT_EQ(batch_results[i].first, single_result.first);
    EXPECT_NEAR(batch_results[i].second, single_result.second, TypeParam(1e-5));
  }
}

TYPED_TEST(ClusterBatchTestT, AssignBatchLargeBatch) {
  constexpr size_t BATCH_SIZE = 10000;

  std::mt19937 gen(789);
  std::uniform_real_distribution<TypeParam> dist(-1.0, 1.0);

  std::vector<TypeParam> embeddings(BATCH_SIZE * this->DIM);
  for (size_t i = 0; i < embeddings.size(); ++i) {
    embeddings[i] = dist(gen);
  }

  auto results = this->engine.assign_batch(embeddings.data(), BATCH_SIZE, this->DIM);

  ASSERT_EQ(results.size(), BATCH_SIZE);
  for (const auto& [cluster_id, distance] : results) {
    EXPECT_GE(cluster_id, 0);
    EXPECT_LT(cluster_id, static_cast<int>(this->N_CLUSTERS));
  }
}

TYPED_TEST(ClusterBatchTestT, AssignBatchSingleItem) {
  std::vector<TypeParam> embedding(this->DIM, TypeParam(0.5));
  
  auto results = this->engine.assign_batch(embedding.data(), 1, this->DIM);
  
  ASSERT_EQ(results.size(), 1);
  auto single_result = this->engine.assign(embedding.data(), this->DIM);
  EXPECT_EQ(results[0].first, single_result.first);
  EXPECT_NEAR(results[0].second, single_result.second, TypeParam(1e-5));
}

TYPED_TEST(ClusterBatchTestT, AssignBatchDeterministic) {
  constexpr size_t BATCH_SIZE = 100;

  std::vector<TypeParam> embeddings(BATCH_SIZE * this->DIM, TypeParam(0.1));

  auto results1 = this->engine.assign_batch(embeddings.data(), BATCH_SIZE, this->DIM);
  auto results2 = this->engine.assign_batch(embeddings.data(), BATCH_SIZE, this->DIM);

  ASSERT_EQ(results1.size(), results2.size());
  for (size_t i = 0; i < results1.size(); ++i) {
    EXPECT_EQ(results1[i].first, results2[i].first);
    EXPECT_EQ(results1[i].second, results2[i].second);
  }
}

// =============================================================================
// SECTION 3: CUDA Backend - Basic Functionality
// =============================================================================

template <typename Scalar> class ClusterEngineCudaTestT : public ::testing::Test {
protected:
  void SetUp() override {
    if (!cuda_available()) {
      GTEST_SKIP() << "CUDA not available, skipping GPU tests";
    }
    engine = ClusterEngine<Scalar>(ClusterBackendType::CUDA);
  }

  ClusterEngine<Scalar> engine{ClusterBackendType::Cpu};

  static void fill_matrix(EmbeddingMatrix<Scalar>& m, std::initializer_list<Scalar> values) {
    auto it = values.begin();
    for (size_t i = 0; i < m.rows(); ++i) {
      for (size_t j = 0; j < m.cols(); ++j) {
        m(i, j) = (it != values.end()) ? *it++ : Scalar(0);
      }
    }
  }

  static void random_matrix(EmbeddingMatrix<Scalar>& m, std::mt19937& gen) {
    std::uniform_real_distribution<Scalar> dist(-1.0, 1.0);
    for (size_t i = 0; i < m.rows(); ++i) {
      for (size_t j = 0; j < m.cols(); ++j) {
        m(i, j) = dist(gen);
      }
    }
  }
};

TYPED_TEST_SUITE(ClusterEngineCudaTestT, ScalarTypes);

TYPED_TEST(ClusterEngineCudaTestT, EmptyEngine) {
  EXPECT_EQ(this->engine.get_n_clusters(), 0);
  EXPECT_TRUE(this->engine.is_gpu_accelerated());
}

TYPED_TEST(ClusterEngineCudaTestT, LoadCentroids) {
  EmbeddingMatrix<TypeParam> centers(3, 4);
  this->fill_matrix(centers, {TypeParam(1.0), TypeParam(0.0), TypeParam(0.0), TypeParam(0.0),
                              TypeParam(0.0), TypeParam(1.0), TypeParam(0.0), TypeParam(0.0),
                              TypeParam(0.0), TypeParam(0.0), TypeParam(1.0), TypeParam(0.0)});

  this->engine.load_centroids(centers);
  EXPECT_EQ(this->engine.get_n_clusters(), 3);
}

TYPED_TEST(ClusterEngineCudaTestT, AssignToNearestCluster) {
  EmbeddingMatrix<TypeParam> centers(3, 4);
  this->fill_matrix(centers, {TypeParam(1.0), TypeParam(0.0), TypeParam(0.0), TypeParam(0.0),
                              TypeParam(0.0), TypeParam(1.0), TypeParam(0.0), TypeParam(0.0),
                              TypeParam(0.0), TypeParam(0.0), TypeParam(1.0), TypeParam(0.0)});

  this->engine.load_centroids(centers);

  TypeParam vec[] = {TypeParam(0.9), TypeParam(0.1), TypeParam(0.0), TypeParam(0.0)};
  auto [cluster_id, distance] = this->engine.assign(vec, 4);
  EXPECT_EQ(cluster_id, 0);
  EXPECT_GT(distance, TypeParam(0.0));
}

TYPED_TEST(ClusterEngineCudaTestT, AssignToSecondCluster) {
  EmbeddingMatrix<TypeParam> centers(3, 4);
  this->fill_matrix(centers, {TypeParam(1.0), TypeParam(0.0), TypeParam(0.0), TypeParam(0.0),
                              TypeParam(0.0), TypeParam(1.0), TypeParam(0.0), TypeParam(0.0),
                              TypeParam(0.0), TypeParam(0.0), TypeParam(1.0), TypeParam(0.0)});

  this->engine.load_centroids(centers);

  TypeParam vec[] = {TypeParam(0.0), TypeParam(0.95), TypeParam(0.05), TypeParam(0.0)};
  auto [cluster_id, distance] = this->engine.assign(vec, 4);
  EXPECT_EQ(cluster_id, 1);
  EXPECT_LT(distance, TypeParam(0.1));
}

TYPED_TEST(ClusterEngineCudaTestT, AssignToThirdCluster) {
  EmbeddingMatrix<TypeParam> centers(3, 4);
  this->fill_matrix(centers, {TypeParam(1.0), TypeParam(0.0), TypeParam(0.0), TypeParam(0.0),
                              TypeParam(0.0), TypeParam(1.0), TypeParam(0.0), TypeParam(0.0),
                              TypeParam(0.0), TypeParam(0.0), TypeParam(1.0), TypeParam(0.0)});

  this->engine.load_centroids(centers);

  TypeParam vec[] = {TypeParam(0.0), TypeParam(0.0), TypeParam(1.0), TypeParam(0.0)};
  auto [cluster_id, distance] = this->engine.assign(vec, 4);
  EXPECT_EQ(cluster_id, 2);
  EXPECT_NEAR(distance, TypeParam(0.0), TypeParam(1e-5));
}

TYPED_TEST(ClusterEngineCudaTestT, ManyClusterAssignment) {
  constexpr size_t N_CLUSTERS = 100;
  constexpr size_t DIM = 128;
  constexpr int N_QUERIES = 50;

  std::mt19937 gen(42);
  EmbeddingMatrix<TypeParam> centers(N_CLUSTERS, DIM);
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

TYPED_TEST(ClusterEngineCudaTestT, SmallDimensionOptimization) {
  constexpr size_t N_CLUSTERS = 10;
  constexpr size_t DIM = 256;

  std::mt19937 gen(42);
  EmbeddingMatrix<TypeParam> centers(N_CLUSTERS, DIM);
  this->random_matrix(centers, gen);

  this->engine.load_centroids(centers);

  std::vector<TypeParam> query(DIM, TypeParam(0.5));
  auto [cluster_id, distance] = this->engine.assign(query.data(), DIM);
  EXPECT_GE(cluster_id, 0);
  EXPECT_LT(cluster_id, static_cast<int>(N_CLUSTERS));
}

TYPED_TEST(ClusterEngineCudaTestT, LargeDimensionOptimization) {
  constexpr size_t N_CLUSTERS = 10;
  constexpr size_t DIM = 1024;

  std::mt19937 gen(42);
  EmbeddingMatrix<TypeParam> centers(N_CLUSTERS, DIM);
  this->random_matrix(centers, gen);

  this->engine.load_centroids(centers);

  std::vector<TypeParam> query(DIM, TypeParam(0.5));
  auto [cluster_id, distance] = this->engine.assign(query.data(), DIM);
  EXPECT_GE(cluster_id, 0);
  EXPECT_LT(cluster_id, static_cast<int>(N_CLUSTERS));
}

TYPED_TEST(ClusterEngineCudaTestT, SmallClusterCountOptimization) {
  constexpr size_t N_CLUSTERS = 50;
  constexpr size_t DIM = 128;

  std::mt19937 gen(42);
  EmbeddingMatrix<TypeParam> centers(N_CLUSTERS, DIM);
  this->random_matrix(centers, gen);

  this->engine.load_centroids(centers);

  std::vector<TypeParam> query(DIM, TypeParam(0.5));
  auto [cluster_id, distance] = this->engine.assign(query.data(), DIM);
  EXPECT_GE(cluster_id, 0);
  EXPECT_LT(cluster_id, static_cast<int>(N_CLUSTERS));
}

TYPED_TEST(ClusterEngineCudaTestT, LargeClusterCountOptimization) {
  constexpr size_t N_CLUSTERS = 200;
  constexpr size_t DIM = 128;

  std::mt19937 gen(42);
  EmbeddingMatrix<TypeParam> centers(N_CLUSTERS, DIM);
  this->random_matrix(centers, gen);

  this->engine.load_centroids(centers);

  std::vector<TypeParam> query(DIM, TypeParam(0.5));
  auto [cluster_id, distance] = this->engine.assign(query.data(), DIM);
  EXPECT_GE(cluster_id, 0);
  EXPECT_LT(cluster_id, static_cast<int>(N_CLUSTERS));
}

TYPED_TEST(ClusterEngineCudaTestT, ReloadCentroidsRecapturesGraph) {
  EmbeddingMatrix<TypeParam> centers1(3, 4);
  this->fill_matrix(centers1, {TypeParam(1.0), TypeParam(0.0), TypeParam(0.0), TypeParam(0.0),
                               TypeParam(0.0), TypeParam(1.0), TypeParam(0.0), TypeParam(0.0),
                               TypeParam(0.0), TypeParam(0.0), TypeParam(1.0), TypeParam(0.0)});

  this->engine.load_centroids(centers1);

  TypeParam vec1[] = {TypeParam(0.9), TypeParam(0.1), TypeParam(0.0), TypeParam(0.0)};
  auto [id1, dist1] = this->engine.assign(vec1, 4);
  EXPECT_EQ(id1, 0);

  EmbeddingMatrix<TypeParam> centers2(3, 4);
  this->fill_matrix(centers2, {TypeParam(0.0), TypeParam(1.0), TypeParam(0.0), TypeParam(0.0),
                               TypeParam(1.0), TypeParam(0.0), TypeParam(0.0), TypeParam(0.0),
                               TypeParam(0.0), TypeParam(0.0), TypeParam(1.0), TypeParam(0.0)});

  this->engine.load_centroids(centers2);

  auto [id2, dist2] = this->engine.assign(vec1, 4);
  EXPECT_EQ(id2, 1);
}

TYPED_TEST(ClusterEngineCudaTestT, DimensionMismatch) {
  EmbeddingMatrix<TypeParam> centers(3, 4);
  this->fill_matrix(centers, {TypeParam(1.0), TypeParam(0.0), TypeParam(0.0), TypeParam(0.0),
                              TypeParam(0.0), TypeParam(1.0), TypeParam(0.0), TypeParam(0.0),
                              TypeParam(0.0), TypeParam(0.0), TypeParam(1.0), TypeParam(0.0)});
  this->engine.load_centroids(centers);

  std::vector<TypeParam> query(3, TypeParam(1.0));
  auto [cluster_id, distance] = this->engine.assign(query.data(), 3);
  EXPECT_EQ(cluster_id, -1);
}

// =============================================================================
// SECTION 4: CUDA Backend - Advanced GPU Tests
// =============================================================================

TEST(CudaAdvancedTest, VeryLargeClusterCount) {
  if (!cuda_available()) GTEST_SKIP();

  constexpr size_t N_CLUSTERS = 5000;
  constexpr size_t DIM = 128;

  ClusterEngine<float> engine(ClusterBackendType::CUDA);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  EmbeddingMatrix<float> centers(N_CLUSTERS, DIM);
  for (size_t i = 0; i < N_CLUSTERS; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
      centers(i, j) = dist(gen);
    }
  }

  engine.load_centroids(centers);
  EXPECT_EQ(engine.get_n_clusters(), static_cast<int>(N_CLUSTERS));

  std::vector<float> query(DIM);
  for (size_t d = 0; d < DIM; ++d) {
    query[d] = dist(gen);
  }

  auto [cluster_id, distance] = engine.assign(query.data(), DIM);
  EXPECT_GE(cluster_id, 0);
  EXPECT_LT(cluster_id, static_cast<int>(N_CLUSTERS));
}

TEST(CudaAdvancedTest, VeryHighDimensionality) {
  if (!cuda_available()) GTEST_SKIP();

  constexpr size_t N_CLUSTERS = 10;
  constexpr size_t DIM = 4096;

  ClusterEngine<float> engine(ClusterBackendType::CUDA);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  EmbeddingMatrix<float> centers(N_CLUSTERS, DIM);
  for (size_t i = 0; i < N_CLUSTERS; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
      centers(i, j) = dist(gen);
    }
  }

  engine.load_centroids(centers);

  std::vector<float> query(DIM);
  for (size_t d = 0; d < DIM; ++d) {
    query[d] = dist(gen);
  }

  auto [cluster_id, distance] = engine.assign(query.data(), DIM);
  EXPECT_GE(cluster_id, 0);
  EXPECT_LT(cluster_id, static_cast<int>(N_CLUSTERS));
}

TEST(CudaAdvancedTest, ArgminWithNonMultipleOfFourClusters) {
  if (!cuda_available()) GTEST_SKIP();

  constexpr size_t N_CLUSTERS = 37;
  constexpr size_t DIM = 64;

  ClusterEngine<float> engine(ClusterBackendType::CUDA);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  EmbeddingMatrix<float> centers(N_CLUSTERS, DIM);
  for (size_t i = 0; i < N_CLUSTERS; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
      centers(i, j) = dist(gen);
    }
  }

  engine.load_centroids(centers);

  std::vector<float> query(DIM);
  for (size_t d = 0; d < DIM; ++d) {
    query[d] = dist(gen);
  }

  auto [cluster_id, distance] = engine.assign(query.data(), DIM);
  EXPECT_GE(cluster_id, 0);
  EXPECT_LT(cluster_id, static_cast<int>(N_CLUSTERS));
}

TEST(CudaAdvancedTest, NormCalculationAccuracy) {
  if (!cuda_available()) GTEST_SKIP();

  constexpr size_t N_CLUSTERS = 10;
  constexpr size_t DIM = 128;

  ClusterEngine<double> engine(ClusterBackendType::CUDA);

  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  EmbeddingMatrix<double> centers(N_CLUSTERS, DIM);
  for (size_t i = 0; i < N_CLUSTERS; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
      centers(i, j) = dist(gen);
    }
  }

  engine.load_centroids(centers);

  std::vector<double> query(DIM);
  for (size_t d = 0; d < DIM; ++d) {
    query[d] = dist(gen);
  }

  auto [cluster_id, distance] = engine.assign(query.data(), DIM);
  EXPECT_GE(cluster_id, 0);
  EXPECT_GE(distance, 0.0);
  EXPECT_FALSE(std::isnan(distance));
  EXPECT_FALSE(std::isinf(distance));
}

TEST(CudaAdvancedTest, MultipleEngineInstances) {
  if (!cuda_available()) GTEST_SKIP();

  constexpr size_t N_CLUSTERS = 20;
  constexpr size_t DIM = 64;

  ClusterEngine<float> engine1(ClusterBackendType::CUDA);
  ClusterEngine<float> engine2(ClusterBackendType::CUDA);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  EmbeddingMatrix<float> centers1(N_CLUSTERS, DIM);
  EmbeddingMatrix<float> centers2(N_CLUSTERS, DIM);
  for (size_t i = 0; i < N_CLUSTERS; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
      centers1(i, j) = dist(gen);
      centers2(i, j) = dist(gen);
    }
  }

  engine1.load_centroids(centers1);
  engine2.load_centroids(centers2);

  std::vector<float> query(DIM);
  for (size_t d = 0; d < DIM; ++d) {
    query[d] = dist(gen);
  }

  auto [id1, dist1] = engine1.assign(query.data(), DIM);
  auto [id2, dist2] = engine2.assign(query.data(), DIM);

  EXPECT_GE(id1, 0);
  EXPECT_GE(id2, 0);
  EXPECT_LT(id1, static_cast<int>(N_CLUSTERS));
  EXPECT_LT(id2, static_cast<int>(N_CLUSTERS));
}

TEST(CudaAdvancedTest, NonMultipleOfFourDimensions) {
  if (!cuda_available()) GTEST_SKIP();

  constexpr size_t N_CLUSTERS = 10;
  constexpr size_t DIM = 127;

  ClusterEngine<float> engine(ClusterBackendType::CUDA);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  EmbeddingMatrix<float> centers(N_CLUSTERS, DIM);
  for (size_t i = 0; i < N_CLUSTERS; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
      centers(i, j) = dist(gen);
    }
  }

  engine.load_centroids(centers);

  std::vector<float> query(DIM);
  for (size_t d = 0; d < DIM; ++d) {
    query[d] = dist(gen);
  }

  auto [cluster_id, distance] = engine.assign(query.data(), DIM);
  EXPECT_GE(cluster_id, 0);
  EXPECT_LT(cluster_id, static_cast<int>(N_CLUSTERS));
}

TEST(CudaAdvancedTest, RepeatedAssignCalls) {
  if (!cuda_available()) GTEST_SKIP();

  constexpr size_t N_CLUSTERS = 50;
  constexpr size_t DIM = 128;
  constexpr int N_ITERATIONS = 1000;

  ClusterEngine<float> engine(ClusterBackendType::CUDA);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  EmbeddingMatrix<float> centers(N_CLUSTERS, DIM);
  for (size_t i = 0; i < N_CLUSTERS; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
      centers(i, j) = dist(gen);
    }
  }

  engine.load_centroids(centers);

  for (int iter = 0; iter < N_ITERATIONS; ++iter) {
    std::vector<float> query(DIM);
    for (size_t d = 0; d < DIM; ++d) {
      query[d] = dist(gen);
    }

    auto [cluster_id, distance] = engine.assign(query.data(), DIM);
    EXPECT_GE(cluster_id, 0);
    EXPECT_LT(cluster_id, static_cast<int>(N_CLUSTERS));
    EXPECT_GE(distance, 0.0f);
  }
}

// =============================================================================
// SECTION 5: CUDA Backend - Thread Safety
// =============================================================================

template <typename Scalar> class ClusterCudaThreadSafetyTestT : public ::testing::Test {
protected:
  static constexpr size_t N_THREADS = 16;
  static constexpr int N_QUERIES_PER_THREAD = 1000;
  static constexpr size_t N_CLUSTERS = 100;
  static constexpr size_t DIM = 128;

  void SetUp() override {
    if (!cuda_available()) {
      GTEST_SKIP() << "CUDA not available, skipping GPU thread safety tests";
    }

    std::mt19937 gen(42);
    std::uniform_real_distribution<Scalar> dist(-1.0, 1.0);

    EmbeddingMatrix<Scalar> centers(N_CLUSTERS, DIM);
    for (size_t i = 0; i < N_CLUSTERS; ++i) {
      for (size_t j = 0; j < DIM; ++j) {
        centers(i, j) = dist(gen);
      }
    }

    engine_ = ClusterEngine<Scalar>(ClusterBackendType::CUDA);
    engine_.load_centroids(centers);
  }

  ClusterEngine<Scalar> engine_{ClusterBackendType::Cpu};
};

TYPED_TEST_SUITE(ClusterCudaThreadSafetyTestT, ScalarTypes);

TYPED_TEST(ClusterCudaThreadSafetyTestT, ConcurrentAssign) {
  std::atomic<int> error_count{0};
  std::atomic<int> success_count{0};
  std::vector<std::thread> threads;

  for (size_t t = 0; t < this->N_THREADS; ++t) {
    threads.emplace_back([&, t]() {
      std::mt19937 gen(static_cast<unsigned>(42 + t));
      std::uniform_real_distribution<TypeParam> dist(-1.0, 1.0);

      for (int q = 0; q < this->N_QUERIES_PER_THREAD; ++q) {
        std::vector<TypeParam> query(this->DIM);
        for (size_t d = 0; d < this->DIM; ++d) {
          query[d] = dist(gen);
        }

        auto [cluster_id, distance] = this->engine_.assign(query.data(), this->DIM);

        if (cluster_id >= 0 && cluster_id < static_cast<int>(this->N_CLUSTERS) && distance >= 0
            && !std::isnan(distance) && !std::isinf(distance)) {
          success_count.fetch_add(1, std::memory_order_relaxed);
        } else {
          error_count.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(error_count.load(), 0) << "Thread safety violations detected in CUDA backend";
  EXPECT_EQ(success_count.load(), static_cast<int>(this->N_THREADS) * this->N_QUERIES_PER_THREAD);
}

TYPED_TEST(ClusterCudaThreadSafetyTestT, StressTest) {
  static constexpr size_t STRESS_THREADS = 64;
  static constexpr int STRESS_QUERIES = 5000;

  std::atomic<int> success_count{0};
  std::atomic<int> failure_count{0};
  std::vector<std::thread> threads;

  for (size_t t = 0; t < STRESS_THREADS; ++t) {
    threads.emplace_back([&, t]() {
      std::mt19937 gen(static_cast<unsigned>(100 + t));
      std::uniform_real_distribution<TypeParam> dist(-1.0, 1.0);

      for (int q = 0; q < STRESS_QUERIES; ++q) {
        std::vector<TypeParam> query(this->DIM);
        for (size_t d = 0; d < this->DIM; ++d) {
          query[d] = dist(gen);
        }

        auto [cluster_id, distance] = this->engine_.assign(query.data(), this->DIM);

        if (cluster_id >= 0 && cluster_id < static_cast<int>(this->N_CLUSTERS) && distance >= 0
            && !std::isnan(distance)) {
          success_count.fetch_add(1, std::memory_order_relaxed);
        } else {
          failure_count.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(failure_count.load(), 0);
  EXPECT_EQ(success_count.load(), static_cast<int>(STRESS_THREADS) * STRESS_QUERIES);
}

// =============================================================================
// SECTION 6: Backend Comparison Tests
// =============================================================================

TEST(BackendComparisonTest, CPUvsCUDAConsistencyFloat) {
  if (!cuda_available()) GTEST_SKIP();

  constexpr size_t N_CLUSTERS = 100;
  constexpr size_t DIM = 128;
  constexpr int N_QUERIES = 100;

  ClusterEngine<float> cpu_engine(ClusterBackendType::Cpu);
  ClusterEngine<float> cuda_engine(ClusterBackendType::CUDA);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  EmbeddingMatrix<float> centers(N_CLUSTERS, DIM);
  for (size_t i = 0; i < N_CLUSTERS; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
      centers(i, j) = dist(gen);
    }
  }

  cpu_engine.load_centroids(centers);
  cuda_engine.load_centroids(centers);

  int mismatch_count = 0;
  for (int q = 0; q < N_QUERIES; ++q) {
    std::vector<float> query(DIM);
    for (size_t d = 0; d < DIM; ++d) {
      query[d] = dist(gen);
    }

    auto [cpu_id, cpu_dist] = cpu_engine.assign(query.data(), DIM);
    auto [cuda_id, cuda_dist] = cuda_engine.assign(query.data(), DIM);

    if (cpu_id != cuda_id) {
      ++mismatch_count;
    }
    EXPECT_NEAR(cpu_dist, cuda_dist, 1e-4f) << "Distance mismatch at query " << q;
  }

  EXPECT_LE(mismatch_count, N_QUERIES / 10) << "Too many cluster assignment mismatches";
}

TEST(BackendComparisonTest, CPUvsCUDAConsistencyDouble) {
  if (!cuda_available()) GTEST_SKIP();

  constexpr size_t N_CLUSTERS = 100;
  constexpr size_t DIM = 128;
  constexpr int N_QUERIES = 100;

  ClusterEngine<double> cpu_engine(ClusterBackendType::Cpu);
  ClusterEngine<double> cuda_engine(ClusterBackendType::CUDA);

  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  EmbeddingMatrix<double> centers(N_CLUSTERS, DIM);
  for (size_t i = 0; i < N_CLUSTERS; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
      centers(i, j) = dist(gen);
    }
  }

  cpu_engine.load_centroids(centers);
  cuda_engine.load_centroids(centers);

  int mismatch_count = 0;
  for (int q = 0; q < N_QUERIES; ++q) {
    std::vector<double> query(DIM);
    for (size_t d = 0; d < DIM; ++d) {
      query[d] = dist(gen);
    }

    auto [cpu_id, cpu_dist] = cpu_engine.assign(query.data(), DIM);
    auto [cuda_id, cuda_dist] = cuda_engine.assign(query.data(), DIM);

    if (cpu_id != cuda_id) {
      ++mismatch_count;
    }
    EXPECT_NEAR(cpu_dist, cuda_dist, 1e-9) << "Distance mismatch at query " << q;
  }

  EXPECT_LE(mismatch_count, N_QUERIES / 20) << "Too many cluster assignment mismatches";
}

TEST(BackendComparisonTest, EdgeCaseConsistency) {
  if (!cuda_available()) GTEST_SKIP();

  constexpr size_t N_CLUSTERS = 10;
  constexpr size_t DIM = 64;

  ClusterEngine<float> cpu_engine(ClusterBackendType::Cpu);
  ClusterEngine<float> cuda_engine(ClusterBackendType::CUDA);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  EmbeddingMatrix<float> centers(N_CLUSTERS, DIM);
  for (size_t i = 0; i < N_CLUSTERS; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
      centers(i, j) = dist(gen);
    }
  }

  cpu_engine.load_centroids(centers);
  cuda_engine.load_centroids(centers);

  std::vector<float> zero_query(DIM, 0.0f);
  auto [cpu_id1, cpu_dist1] = cpu_engine.assign(zero_query.data(), DIM);
  auto [cuda_id1, cuda_dist1] = cuda_engine.assign(zero_query.data(), DIM);
  EXPECT_EQ(cpu_id1, cuda_id1) << "Zero embedding mismatch";
  EXPECT_NEAR(cpu_dist1, cuda_dist1, 1e-4f);

  std::vector<float> small_query(DIM, 1e-7f);
  auto [cpu_id2, cpu_dist2] = cpu_engine.assign(small_query.data(), DIM);
  auto [cuda_id2, cuda_dist2] = cuda_engine.assign(small_query.data(), DIM);
  EXPECT_EQ(cpu_id2, cuda_id2) << "Small values mismatch";
  EXPECT_NEAR(cpu_dist2, cuda_dist2, 1e-4f);

  std::vector<float> large_query(DIM, 1e6f);
  auto [cpu_id3, cpu_dist3] = cpu_engine.assign(large_query.data(), DIM);
  auto [cuda_id3, cuda_dist3] = cuda_engine.assign(large_query.data(), DIM);
  EXPECT_EQ(cpu_id3, cuda_id3) << "Large values mismatch";
}

// =============================================================================
// SECTION 7: Backend Factory & Error Handling
// =============================================================================

TEST(ClusterBackendFactoryTest, AutoSelectsCUDAWhenAvailable) {
  auto engine = ClusterEngine<float>(ClusterBackendType::Auto);

  if (cuda_available()) {
    EXPECT_TRUE(engine.is_gpu_accelerated());
  } else {
    EXPECT_FALSE(engine.is_gpu_accelerated());
  }
}

TEST(ClusterBackendFactoryTest, ExplicitCPUBackend) {
  auto engine = ClusterEngine<float>(ClusterBackendType::Cpu);
  EXPECT_FALSE(engine.is_gpu_accelerated());
}

TEST(ClusterBackendFactoryTest, ExplicitCUDABackendFallback) {
  auto engine = ClusterEngine<float>(ClusterBackendType::CUDA);

  if (!cuda_available()) {
    EXPECT_FALSE(engine.is_gpu_accelerated());
  } else {
    EXPECT_TRUE(engine.is_gpu_accelerated());
  }
}

TEST(ClusterBackendFactoryTest, CudaAvailableFunction) {
  bool available = cuda_available();
  (void)available;
}

TEST(ClusterBackendFactoryTest, DoubleSupport) {
  ClusterEngine<double> cpu_engine(ClusterBackendType::Cpu);
  EXPECT_FALSE(cpu_engine.is_gpu_accelerated());

  if (cuda_available()) {
    ClusterEngine<double> cuda_engine(ClusterBackendType::CUDA);
    EXPECT_TRUE(cuda_engine.is_gpu_accelerated());
  }
}

// =============================================================================
// SECTION 8: Edge Cases & Robustness
// =============================================================================

TEST(ClusterEdgeCasesTest, ZeroClusters) {
  ClusterEngine<float> engine(ClusterBackendType::Cpu);
  std::vector<float> query(128, 1.0f);
  auto [id, dist] = engine.assign(query.data(), 128);
  EXPECT_EQ(id, -1);
}

TEST(ClusterEdgeCasesTest, SingleCluster) {
  ClusterEngine<float> engine(ClusterBackendType::Cpu);

  EmbeddingMatrix<float> centers(1, 4);
  centers(0, 0) = 1.0f;
  centers(0, 1) = 0.0f;
  centers(0, 2) = 0.0f;
  centers(0, 3) = 0.0f;

  engine.load_centroids(centers);

  std::vector<float> query1{0.9f, 0.1f, 0.0f, 0.0f};
  auto [id1, dist1] = engine.assign(query1.data(), 4);
  EXPECT_EQ(id1, 0);

  std::vector<float> query2{0.0f, 0.0f, 1.0f, 0.0f};
  auto [id2, dist2] = engine.assign(query2.data(), 4);
  EXPECT_EQ(id2, 0);
}

TEST(ClusterEdgeCasesTest, TwoClusters) {
  ClusterEngine<float> engine(ClusterBackendType::Cpu);

  EmbeddingMatrix<float> centers(2, 2);
  centers(0, 0) = 1.0f;
  centers(0, 1) = 0.0f;
  centers(1, 0) = 0.0f;
  centers(1, 1) = 1.0f;

  engine.load_centroids(centers);

  std::vector<float> query1{0.9f, 0.1f};
  auto [id1, dist1] = engine.assign(query1.data(), 2);
  EXPECT_EQ(id1, 0);

  std::vector<float> query2{0.1f, 0.9f};
  auto [id2, dist2] = engine.assign(query2.data(), 2);
  EXPECT_EQ(id2, 1);
}

TEST(ClusterEdgeCasesTest, EmbeddingAllZeros) {
  ClusterEngine<float> engine(ClusterBackendType::Cpu);

  EmbeddingMatrix<float> centers(3, 4);
  centers(0, 0) = 1.0f;
  centers(0, 1) = 0.0f;
  centers(0, 2) = 0.0f;
  centers(0, 3) = 0.0f;
  centers(1, 0) = 0.0f;
  centers(1, 1) = 1.0f;
  centers(1, 2) = 0.0f;
  centers(1, 3) = 0.0f;
  centers(2, 0) = 0.0f;
  centers(2, 1) = 0.0f;
  centers(2, 2) = 1.0f;
  centers(2, 3) = 0.0f;

  engine.load_centroids(centers);

  std::vector<float> zero_query(4, 0.0f);
  auto [id, dist] = engine.assign(zero_query.data(), 4);
  EXPECT_GE(id, 0);
  EXPECT_LT(id, 3);
  EXPECT_GE(dist, 0.0f);
}

TEST(ClusterEdgeCasesTest, VerySmallValues) {
  ClusterEngine<float> engine(ClusterBackendType::Cpu);

  EmbeddingMatrix<float> centers(2, 4);
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      centers(i, j) = 1e-7f * static_cast<float>(i + j);
    }
  }

  engine.load_centroids(centers);

  std::vector<float> query(4, 1e-7f);
  auto [id, dist] = engine.assign(query.data(), 4);
  EXPECT_GE(id, 0);
  EXPECT_LT(id, 2);
}

TEST(ClusterEdgeCasesTest, VeryLargeValues) {
  ClusterEngine<float> engine(ClusterBackendType::Cpu);

  EmbeddingMatrix<float> centers(2, 4);
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      centers(i, j) = 1e6f * static_cast<float>(i + j + 1);
    }
  }

  engine.load_centroids(centers);

  std::vector<float> query(4, 1e6f);
  auto [id, dist] = engine.assign(query.data(), 4);
  EXPECT_GE(id, 0);
  EXPECT_LT(id, 2);
}

TEST(ClusterEdgeCasesTest, MixedScaleValues) {
  ClusterEngine<float> engine(ClusterBackendType::Cpu);

  EmbeddingMatrix<float> centers(2, 4);
  centers(0, 0) = 1e-7f;
  centers(0, 1) = 1e7f;
  centers(0, 2) = 1e-3f;
  centers(0, 3) = 1e3f;
  centers(1, 0) = 1e7f;
  centers(1, 1) = 1e-7f;
  centers(1, 2) = 1e3f;
  centers(1, 3) = 1e-3f;

  engine.load_centroids(centers);

  std::vector<float> query{1e-6f, 1e6f, 1e-2f, 1e2f};
  auto [id, dist] = engine.assign(query.data(), 4);
  EXPECT_GE(id, 0);
  EXPECT_LT(id, 2);
}

TEST(ClusterEdgeCasesTest, IdenticalCentroids) {
  ClusterEngine<float> engine(ClusterBackendType::Cpu);

  EmbeddingMatrix<float> centers(3, 4);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      centers(i, j) = 1.0f;
    }
  }

  engine.load_centroids(centers);

  std::vector<float> query{1.0f, 1.0f, 1.0f, 1.0f};
  auto [id, dist] = engine.assign(query.data(), 4);
  EXPECT_GE(id, 0);
  EXPECT_LT(id, 3);
  EXPECT_NEAR(dist, 0.0f, 1e-5f);
}

TEST(ClusterEdgeCasesTest, CUDAZeroClusters) {
  if (!cuda_available()) GTEST_SKIP();

  ClusterEngine<float> engine(ClusterBackendType::CUDA);
  std::vector<float> query(128, 1.0f);
  auto [id, dist] = engine.assign(query.data(), 128);
  EXPECT_EQ(id, -1);
}

TEST(ClusterEdgeCasesTest, CUDASingleCluster) {
  if (!cuda_available()) GTEST_SKIP();

  ClusterEngine<float> engine(ClusterBackendType::CUDA);

  EmbeddingMatrix<float> centers(1, 4);
  centers(0, 0) = 1.0f;
  centers(0, 1) = 0.0f;
  centers(0, 2) = 0.0f;
  centers(0, 3) = 0.0f;

  engine.load_centroids(centers);

  std::vector<float> query{0.5f, 0.5f, 0.5f, 0.5f};
  auto [id, dist] = engine.assign(query.data(), 4);
  EXPECT_EQ(id, 0);
  EXPECT_GE(dist, 0.0f);
}
