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
  EmbeddingMatrixT<TypeParam> centers(N_CLUSTERS, DIM);
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
  EmbeddingMatrixT<TypeParam> centers(N_CLUSTERS, DIM);
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

    EmbeddingMatrixT<Scalar> centers(N_CLUSTERS, DIM);
    for (size_t i = 0; i < N_CLUSTERS; ++i) {
      for (size_t j = 0; j < DIM; ++j) {
        centers(i, j) = dist(gen);
      }
    }

    engine_.load_centroids(centers);
  }

  ClusterEngineT<Scalar> engine_{ClusterBackendType::Cpu};
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
// SECTION 3: CUDA Backend - Basic Functionality
// =============================================================================

template <typename Scalar> class ClusterEngineCudaTestT : public ::testing::Test {
protected:
  void SetUp() override {
    if (!cuda_available()) {
      GTEST_SKIP() << "CUDA not available, skipping GPU tests";
    }
    engine = ClusterEngineT<Scalar>(ClusterBackendType::CUDA);
  }

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

TYPED_TEST_SUITE(ClusterEngineCudaTestT, ScalarTypes);

TYPED_TEST(ClusterEngineCudaTestT, EmptyEngine) {
  EXPECT_EQ(this->engine.get_n_clusters(), 0);
  EXPECT_TRUE(this->engine.is_gpu_accelerated());
}

TYPED_TEST(ClusterEngineCudaTestT, LoadCentroids) {
  EmbeddingMatrixT<TypeParam> centers(3, 4);
  this->fill_matrix(centers, {TypeParam(1.0), TypeParam(0.0), TypeParam(0.0), TypeParam(0.0),
                              TypeParam(0.0), TypeParam(1.0), TypeParam(0.0), TypeParam(0.0),
                              TypeParam(0.0), TypeParam(0.0), TypeParam(1.0), TypeParam(0.0)});

  this->engine.load_centroids(centers);
  EXPECT_EQ(this->engine.get_n_clusters(), 3);
}

TYPED_TEST(ClusterEngineCudaTestT, AssignToNearestCluster) {
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

TYPED_TEST(ClusterEngineCudaTestT, AssignToSecondCluster) {
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

TYPED_TEST(ClusterEngineCudaTestT, AssignToThirdCluster) {
  EmbeddingMatrixT<TypeParam> centers(3, 4);
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

TYPED_TEST(ClusterEngineCudaTestT, SmallDimensionOptimization) {
  constexpr size_t N_CLUSTERS = 10;
  constexpr size_t DIM = 256;

  std::mt19937 gen(42);
  EmbeddingMatrixT<TypeParam> centers(N_CLUSTERS, DIM);
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
  EmbeddingMatrixT<TypeParam> centers(N_CLUSTERS, DIM);
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
  EmbeddingMatrixT<TypeParam> centers(N_CLUSTERS, DIM);
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
  EmbeddingMatrixT<TypeParam> centers(N_CLUSTERS, DIM);
  this->random_matrix(centers, gen);

  this->engine.load_centroids(centers);

  std::vector<TypeParam> query(DIM, TypeParam(0.5));
  auto [cluster_id, distance] = this->engine.assign(query.data(), DIM);
  EXPECT_GE(cluster_id, 0);
  EXPECT_LT(cluster_id, static_cast<int>(N_CLUSTERS));
}

TYPED_TEST(ClusterEngineCudaTestT, ReloadCentroidsRecapturesGraph) {
  EmbeddingMatrixT<TypeParam> centers1(3, 4);
  this->fill_matrix(centers1, {TypeParam(1.0), TypeParam(0.0), TypeParam(0.0), TypeParam(0.0),
                               TypeParam(0.0), TypeParam(1.0), TypeParam(0.0), TypeParam(0.0),
                               TypeParam(0.0), TypeParam(0.0), TypeParam(1.0), TypeParam(0.0)});

  this->engine.load_centroids(centers1);

  TypeParam vec1[] = {TypeParam(0.9), TypeParam(0.1), TypeParam(0.0), TypeParam(0.0)};
  auto [id1, dist1] = this->engine.assign(vec1, 4);
  EXPECT_EQ(id1, 0);

  EmbeddingMatrixT<TypeParam> centers2(3, 4);
  this->fill_matrix(centers2, {TypeParam(0.0), TypeParam(1.0), TypeParam(0.0), TypeParam(0.0),
                               TypeParam(1.0), TypeParam(0.0), TypeParam(0.0), TypeParam(0.0),
                               TypeParam(0.0), TypeParam(0.0), TypeParam(1.0), TypeParam(0.0)});

  this->engine.load_centroids(centers2);

  auto [id2, dist2] = this->engine.assign(vec1, 4);
  EXPECT_EQ(id2, 1);
}

TYPED_TEST(ClusterEngineCudaTestT, DimensionMismatch) {
  EmbeddingMatrixT<TypeParam> centers(3, 4);
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

  ClusterEngineT<float> engine(ClusterBackendType::CUDA);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  EmbeddingMatrixT<float> centers(N_CLUSTERS, DIM);
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

  ClusterEngineT<float> engine(ClusterBackendType::CUDA);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  EmbeddingMatrixT<float> centers(N_CLUSTERS, DIM);
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

  ClusterEngineT<float> engine(ClusterBackendType::CUDA);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  EmbeddingMatrixT<float> centers(N_CLUSTERS, DIM);
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

  ClusterEngineT<double> engine(ClusterBackendType::CUDA);

  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  EmbeddingMatrixT<double> centers(N_CLUSTERS, DIM);
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

  ClusterEngineT<float> engine1(ClusterBackendType::CUDA);
  ClusterEngineT<float> engine2(ClusterBackendType::CUDA);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  EmbeddingMatrixT<float> centers1(N_CLUSTERS, DIM);
  EmbeddingMatrixT<float> centers2(N_CLUSTERS, DIM);
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

  ClusterEngineT<float> engine(ClusterBackendType::CUDA);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  EmbeddingMatrixT<float> centers(N_CLUSTERS, DIM);
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

  ClusterEngineT<float> engine(ClusterBackendType::CUDA);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  EmbeddingMatrixT<float> centers(N_CLUSTERS, DIM);
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

    EmbeddingMatrixT<Scalar> centers(N_CLUSTERS, DIM);
    for (size_t i = 0; i < N_CLUSTERS; ++i) {
      for (size_t j = 0; j < DIM; ++j) {
        centers(i, j) = dist(gen);
      }
    }

    engine_ = ClusterEngineT<Scalar>(ClusterBackendType::CUDA);
    engine_.load_centroids(centers);
  }

  ClusterEngineT<Scalar> engine_{ClusterBackendType::Cpu};
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

  ClusterEngineT<float> cpu_engine(ClusterBackendType::Cpu);
  ClusterEngineT<float> cuda_engine(ClusterBackendType::CUDA);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  EmbeddingMatrixT<float> centers(N_CLUSTERS, DIM);
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

  ClusterEngineT<double> cpu_engine(ClusterBackendType::Cpu);
  ClusterEngineT<double> cuda_engine(ClusterBackendType::CUDA);

  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  EmbeddingMatrixT<double> centers(N_CLUSTERS, DIM);
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

  ClusterEngineT<float> cpu_engine(ClusterBackendType::Cpu);
  ClusterEngineT<float> cuda_engine(ClusterBackendType::CUDA);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  EmbeddingMatrixT<float> centers(N_CLUSTERS, DIM);
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
  auto engine = ClusterEngineT<float>(ClusterBackendType::Auto);

  if (cuda_available()) {
    EXPECT_TRUE(engine.is_gpu_accelerated());
  } else {
    EXPECT_FALSE(engine.is_gpu_accelerated());
  }
}

TEST(ClusterBackendFactoryTest, ExplicitCPUBackend) {
  auto engine = ClusterEngineT<float>(ClusterBackendType::Cpu);
  EXPECT_FALSE(engine.is_gpu_accelerated());
}

TEST(ClusterBackendFactoryTest, ExplicitCUDABackendFallback) {
  auto engine = ClusterEngineT<float>(ClusterBackendType::CUDA);

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
  ClusterEngineT<double> cpu_engine(ClusterBackendType::Cpu);
  EXPECT_FALSE(cpu_engine.is_gpu_accelerated());

  if (cuda_available()) {
    ClusterEngineT<double> cuda_engine(ClusterBackendType::CUDA);
    EXPECT_TRUE(cuda_engine.is_gpu_accelerated());
  }
}

// =============================================================================
// SECTION 8: Edge Cases & Robustness
// =============================================================================

TEST(ClusterEdgeCasesTest, ZeroClusters) {
  ClusterEngineT<float> engine(ClusterBackendType::Cpu);
  std::vector<float> query(128, 1.0f);
  auto [id, dist] = engine.assign(query.data(), 128);
  EXPECT_EQ(id, -1);
}

TEST(ClusterEdgeCasesTest, SingleCluster) {
  ClusterEngineT<float> engine(ClusterBackendType::Cpu);

  EmbeddingMatrixT<float> centers(1, 4);
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
  ClusterEngineT<float> engine(ClusterBackendType::Cpu);

  EmbeddingMatrixT<float> centers(2, 2);
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
  ClusterEngineT<float> engine(ClusterBackendType::Cpu);

  EmbeddingMatrixT<float> centers(3, 4);
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
  ClusterEngineT<float> engine(ClusterBackendType::Cpu);

  EmbeddingMatrixT<float> centers(2, 4);
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
  ClusterEngineT<float> engine(ClusterBackendType::Cpu);

  EmbeddingMatrixT<float> centers(2, 4);
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
  ClusterEngineT<float> engine(ClusterBackendType::Cpu);

  EmbeddingMatrixT<float> centers(2, 4);
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
  ClusterEngineT<float> engine(ClusterBackendType::Cpu);

  EmbeddingMatrixT<float> centers(3, 4);
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

  ClusterEngineT<float> engine(ClusterBackendType::CUDA);
  std::vector<float> query(128, 1.0f);
  auto [id, dist] = engine.assign(query.data(), 128);
  EXPECT_EQ(id, -1);
}

TEST(ClusterEdgeCasesTest, CUDASingleCluster) {
  if (!cuda_available()) GTEST_SKIP();

  ClusterEngineT<float> engine(ClusterBackendType::CUDA);

  EmbeddingMatrixT<float> centers(1, 4);
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
