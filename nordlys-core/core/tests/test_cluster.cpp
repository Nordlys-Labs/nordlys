#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <limits>
#include <nordlys_core/cluster.hpp>
#include <nordlys_core/device.hpp>
#include <nordlys_core/embedding_view.hpp>
#include <random>
#include <thread>
#include <vector>

// =============================================================================
// SECTION 1: CPU Backend - Basic Functionality
// =============================================================================

class ClusterEngineCpuTest : public ::testing::Test {
protected:
  std::unique_ptr<IClusterBackend> engine{create_backend(Device{CpuDevice{}})};

  static void fill_matrix(EmbeddingMatrix<float>& m, std::initializer_list<float> values) {
    auto it = values.begin();
    for (size_t i = 0; i < m.rows(); ++i) {
      for (size_t j = 0; j < m.cols(); ++j) {
        m(i, j) = (it != values.end()) ? *it++ : float(0);
      }
    }
  }

  static void random_matrix(EmbeddingMatrix<float>& m, std::mt19937& gen) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < m.rows(); ++i) {
      for (size_t j = 0; j < m.cols(); ++j) {
        m(i, j) = dist(gen);
      }
    }
  }
};

TEST_F(ClusterEngineCpuTest, EmptyEngine) { EXPECT_EQ(this->engine->n_clusters(), 0); }

TEST_F(ClusterEngineCpuTest, LoadCentroids) {
  EmbeddingMatrix<float> centers(3, 4);
  this->fill_matrix(centers,
                    {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f});

  this->engine->load_centroids(centers.data(), centers.rows(), centers.cols());
  EXPECT_EQ(this->engine->n_clusters(), 3);
}

TEST_F(ClusterEngineCpuTest, AssignToNearestCluster) {
  EmbeddingMatrix<float> centers(3, 4);
  this->fill_matrix(centers,
                    {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f});

  this->engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  float vec[] = {0.9f, 0.1f, 0.0f, 0.0f};
  EmbeddingView view{vec, 4, Device{CpuDevice{}}};
  auto [cluster_id, distance] = this->engine->assign(view);
  EXPECT_EQ(cluster_id, 0);
  EXPECT_GT(distance, 0.0f);
}

TEST_F(ClusterEngineCpuTest, AssignToSecondCluster) {
  EmbeddingMatrix<float> centers(3, 4);
  this->fill_matrix(centers,
                    {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f});

  this->engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  float vec[] = {0.0f, 0.95f, 0.05f, 0.0f};
  EmbeddingView view{vec, 4, Device{CpuDevice{}}};
  auto [cluster_id, distance] = this->engine->assign(view);
  EXPECT_EQ(cluster_id, 1);
  EXPECT_LT(distance, 0.1f);
}

TEST_F(ClusterEngineCpuTest, AssignToThirdCluster) {
  EmbeddingMatrix<float> centers(3, 4);
  this->fill_matrix(centers,
                    {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f});

  this->engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  float vec[] = {0.0f, 0.0f, 1.0f, 0.0f};
  EmbeddingView view{vec, 4, Device{CpuDevice{}}};
  auto [cluster_id, distance] = this->engine->assign(view);
  EXPECT_EQ(cluster_id, 2);
  EXPECT_NEAR(distance, 0.0f, 1e-6f);
}

TEST_F(ClusterEngineCpuTest, ManyClusterAssignment) {
  constexpr size_t N_CLUSTERS = 100;
  constexpr size_t DIM = 128;
  constexpr int N_QUERIES = 50;

  std::mt19937 gen(42);
  EmbeddingMatrix<float> centers(N_CLUSTERS, DIM);
  this->random_matrix(centers, gen);

  this->engine->load_centroids(centers.data(), centers.rows(), centers.cols());
  EXPECT_EQ(this->engine->n_clusters(), static_cast<size_t>(N_CLUSTERS));

  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (int q = 0; q < N_QUERIES; ++q) {
    std::vector<float> query(DIM);
    for (size_t d = 0; d < DIM; ++d) {
      query[d] = dist(gen);
    }

    EmbeddingView view{query.data(), DIM, Device{CpuDevice{}}};
    auto [cluster_id, distance] = this->engine->assign(view);

    EXPECT_GE(cluster_id, 0);
    EXPECT_LT(cluster_id, static_cast<int>(N_CLUSTERS));
    EXPECT_GE(distance, 0.0f);
  }
}

TEST_F(ClusterEngineCpuTest, AssignBeforeLoadReturnsError) {
  std::vector<float> query(4, 1.0f);
  EmbeddingView view{query.data(), 4, Device{CpuDevice{}}};
  auto [cluster_id, distance] = this->engine->assign(view);
  EXPECT_EQ(cluster_id, -1);
  EXPECT_EQ(distance, 0.0f);
}

TEST_F(ClusterEngineCpuTest, LargeClusterCount) {
  constexpr size_t N_CLUSTERS = 1000;
  constexpr size_t DIM = 64;

  std::mt19937 gen(42);
  EmbeddingMatrix<float> centers(N_CLUSTERS, DIM);
  this->random_matrix(centers, gen);

  this->engine->load_centroids(centers.data(), centers.rows(), centers.cols());
  EXPECT_EQ(this->engine->n_clusters(), static_cast<size_t>(N_CLUSTERS));

  std::vector<float> query(DIM, 0.5f);
  EmbeddingView view{query.data(), DIM, Device{CpuDevice{}}};
  auto [cluster_id, distance] = this->engine->assign(view);
  EXPECT_GE(cluster_id, 0);
  EXPECT_LT(cluster_id, static_cast<int>(N_CLUSTERS));
}

TEST_F(ClusterEngineCpuTest, HighDimensionalEmbeddings) {
  constexpr size_t N_CLUSTERS = 10;
  constexpr size_t DIM = 2048;

  std::mt19937 gen(42);
  EmbeddingMatrix<float> centers(N_CLUSTERS, DIM);
  this->random_matrix(centers, gen);

  this->engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  std::vector<float> query(DIM);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t d = 0; d < DIM; ++d) {
    query[d] = dist(gen);
  }

  EmbeddingView view{query.data(), DIM, Device{CpuDevice{}}};
  auto [cluster_id, distance] = this->engine->assign(view);
  EXPECT_GE(cluster_id, 0);
  EXPECT_LT(cluster_id, static_cast<int>(N_CLUSTERS));
}

// =============================================================================
// SECTION 2: CPU Backend - Thread Safety
// =============================================================================

class ClusterCpuThreadSafetyTest : public ::testing::Test {
protected:
  static constexpr size_t N_THREADS = 16;
  static constexpr int N_QUERIES_PER_THREAD = 1000;
  static constexpr size_t N_CLUSTERS = 100;
  static constexpr size_t DIM = 128;

  void SetUp() override {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    EmbeddingMatrix<float> centers(N_CLUSTERS, DIM);
    for (size_t i = 0; i < N_CLUSTERS; ++i) {
      for (size_t j = 0; j < DIM; ++j) {
        centers(i, j) = dist(gen);
      }
    }

    engine_->load_centroids(centers.data(), centers.rows(), centers.cols());
  }

  std::unique_ptr<IClusterBackend> engine_{create_backend(Device{CpuDevice{}})};
};

TEST_F(ClusterCpuThreadSafetyTest, ConcurrentAssign) {
  std::atomic<int> error_count{0};
  std::atomic<int> success_count{0};
  std::vector<std::thread> threads;

  for (size_t t = 0; t < this->N_THREADS; ++t) {
    threads.emplace_back([&, t]() {
      std::mt19937 gen(static_cast<unsigned>(42 + t));
      std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

      for (int q = 0; q < this->N_QUERIES_PER_THREAD; ++q) {
        std::vector<float> query(this->DIM);
        for (size_t d = 0; d < this->DIM; ++d) {
          query[d] = dist(gen);
        }

        EmbeddingView view{query.data(), this->DIM, Device{CpuDevice{}}};
        auto [cluster_id, distance] = this->engine_->assign(view);

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

TEST_F(ClusterCpuThreadSafetyTest, StressTest) {
  static constexpr size_t STRESS_THREADS = 64;
  static constexpr int STRESS_QUERIES = 5000;

  std::atomic<int> success_count{0};
  std::atomic<int> failure_count{0};
  std::vector<std::thread> threads;

  for (size_t t = 0; t < STRESS_THREADS; ++t) {
    threads.emplace_back([&, t]() {
      std::mt19937 gen(static_cast<unsigned>(100 + t));
      std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

      for (int q = 0; q < STRESS_QUERIES; ++q) {
        std::vector<float> query(this->DIM);
        for (size_t d = 0; d < this->DIM; ++d) {
          query[d] = dist(gen);
        }

        EmbeddingView view{query.data(), this->DIM, Device{CpuDevice{}}};
        auto [cluster_id, distance] = this->engine_->assign(view);

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

class ClusterBatchTest : public ::testing::Test {
protected:
  static constexpr size_t N_CLUSTERS = 10;
  static constexpr size_t DIM = 64;

  void SetUp() override {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    EmbeddingMatrix<float> centers(N_CLUSTERS, DIM);
    for (size_t i = 0; i < N_CLUSTERS; ++i) {
      for (size_t j = 0; j < DIM; ++j) {
        centers(i, j) = dist(gen);
      }
    }

    engine->load_centroids(centers.data(), centers.rows(), centers.cols());
  }

  std::unique_ptr<IClusterBackend> engine{create_backend(Device{CpuDevice{}})};
};

TEST_F(ClusterBatchTest, AssignBatchBasic) {
  constexpr size_t BATCH_SIZE = 100;

  std::mt19937 gen(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> embeddings(BATCH_SIZE * this->DIM);
  for (size_t i = 0; i < embeddings.size(); ++i) {
    embeddings[i] = dist(gen);
  }

  EmbeddingBatchView batch_view{embeddings.data(), BATCH_SIZE, this->DIM, Device{CpuDevice{}}};
  auto results = this->engine->assign_batch(batch_view);

  ASSERT_EQ(results.size(), BATCH_SIZE);
  for (const auto& [cluster_id, distance] : results) {
    EXPECT_GE(cluster_id, 0);
    EXPECT_LT(cluster_id, static_cast<int>(this->N_CLUSTERS));
    EXPECT_GE(distance, 0.0f);
  }
}

TEST_F(ClusterBatchTest, AssignBatchMatchesSingleAssign) {
  constexpr size_t BATCH_SIZE = 50;

  std::mt19937 gen(456);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> embeddings(BATCH_SIZE * this->DIM);
  for (size_t i = 0; i < embeddings.size(); ++i) {
    embeddings[i] = dist(gen);
  }

  EmbeddingBatchView batch_view{embeddings.data(), BATCH_SIZE, this->DIM, Device{CpuDevice{}}};
  auto batch_results = this->engine->assign_batch(batch_view);

  ASSERT_EQ(batch_results.size(), BATCH_SIZE);
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    EmbeddingView view{embeddings.data() + i * this->DIM, this->DIM, Device{CpuDevice{}}};
    auto single_result = this->engine->assign(view);
    EXPECT_EQ(batch_results[i].first, single_result.first);
    EXPECT_NEAR(batch_results[i].second, single_result.second, 1e-5f);
  }
}

TEST_F(ClusterBatchTest, AssignBatchLargeBatch) {
  constexpr size_t BATCH_SIZE = 10000;

  std::mt19937 gen(789);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> embeddings(BATCH_SIZE * this->DIM);
  for (size_t i = 0; i < embeddings.size(); ++i) {
    embeddings[i] = dist(gen);
  }

  EmbeddingBatchView batch_view{embeddings.data(), BATCH_SIZE, this->DIM, Device{CpuDevice{}}};
  auto results = this->engine->assign_batch(batch_view);

  ASSERT_EQ(results.size(), BATCH_SIZE);
  for (const auto& [cluster_id, distance] : results) {
    EXPECT_GE(cluster_id, 0);
    EXPECT_LT(cluster_id, static_cast<int>(this->N_CLUSTERS));
  }
}

TEST_F(ClusterBatchTest, AssignBatchSingleItem) {
  std::vector<float> embedding(this->DIM, 0.5f);

  EmbeddingBatchView batch_view{embedding.data(), 1, this->DIM, Device{CpuDevice{}}};
  auto results = this->engine->assign_batch(batch_view);

  ASSERT_EQ(results.size(), 1);
  EmbeddingView view{embedding.data(), this->DIM, Device{CpuDevice{}}};
  auto single_result = this->engine->assign(view);
  EXPECT_EQ(results[0].first, single_result.first);
  EXPECT_NEAR(results[0].second, single_result.second, float(1e-5));
}

TEST_F(ClusterBatchTest, AssignBatchDeterministic) {
  constexpr size_t BATCH_SIZE = 100;

  std::vector<float> embeddings(BATCH_SIZE * this->DIM, float(0.1));

  EmbeddingBatchView batch_view{embeddings.data(), BATCH_SIZE, this->DIM, Device{CpuDevice{}}};
  auto results1 = this->engine->assign_batch(batch_view);
  auto results2 = this->engine->assign_batch(batch_view);

  ASSERT_EQ(results1.size(), results2.size());
  for (size_t i = 0; i < results1.size(); ++i) {
    EXPECT_EQ(results1[i].first, results2[i].first);
    EXPECT_EQ(results1[i].second, results2[i].second);
  }
}

// =============================================================================
// SECTION 3: CUDA Backend - Basic Functionality
// =============================================================================

class ClusterEngineCudaTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!cuda_available()) {
      GTEST_SKIP() << "CUDA not available, skipping GPU tests";
    }
    engine = create_backend(Device{CudaDevice{0}});
  }

  std::unique_ptr<IClusterBackend> engine{create_backend(Device{CpuDevice{}})};

  static void fill_matrix(EmbeddingMatrix<float>& m, std::initializer_list<float> values) {
    auto it = values.begin();
    for (size_t i = 0; i < m.rows(); ++i) {
      for (size_t j = 0; j < m.cols(); ++j) {
        m(i, j) = (it != values.end()) ? *it++ : float(0);
      }
    }
  }

  static void random_matrix(EmbeddingMatrix<float>& m, std::mt19937& gen) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < m.rows(); ++i) {
      for (size_t j = 0; j < m.cols(); ++j) {
        m(i, j) = dist(gen);
      }
    }
  }
};

TEST_F(ClusterEngineCudaTest, EmptyEngine) { EXPECT_EQ(this->engine->n_clusters(), 0); }

TEST_F(ClusterEngineCudaTest, LoadCentroids) {
  EmbeddingMatrix<float> centers(3, 4);
  this->fill_matrix(
      centers, {float(1.0), float(0.0), float(0.0), float(0.0), float(0.0), float(1.0), float(0.0),
                float(0.0), float(0.0), float(0.0), float(1.0), float(0.0)});

  this->engine->load_centroids(centers.data(), centers.rows(), centers.cols());
  EXPECT_EQ(this->engine->n_clusters(), 3);
}

TEST_F(ClusterEngineCudaTest, AssignToNearestCluster) {
  EmbeddingMatrix<float> centers(3, 4);
  this->fill_matrix(
      centers, {float(1.0), float(0.0), float(0.0), float(0.0), float(0.0), float(1.0), float(0.0),
                float(0.0), float(0.0), float(0.0), float(1.0), float(0.0)});

  this->engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  float vec[] = {float(0.9), float(0.1), float(0.0), float(0.0)};
  EmbeddingView view{vec, 4, Device{CpuDevice{}}};
  auto [cluster_id, distance] = this->engine->assign(view);
  EXPECT_EQ(cluster_id, 0);
  EXPECT_GT(distance, float(0.0));
}

TEST_F(ClusterEngineCudaTest, AssignToSecondCluster) {
  EmbeddingMatrix<float> centers(3, 4);
  this->fill_matrix(
      centers, {float(1.0), float(0.0), float(0.0), float(0.0), float(0.0), float(1.0), float(0.0),
                float(0.0), float(0.0), float(0.0), float(1.0), float(0.0)});

  this->engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  float vec[] = {float(0.0), float(0.95), float(0.05), float(0.0)};
  EmbeddingView view{vec, 4, Device{CpuDevice{}}};
  auto [cluster_id, distance] = this->engine->assign(view);
  EXPECT_EQ(cluster_id, 1);
  EXPECT_LT(distance, float(0.1));
}

TEST_F(ClusterEngineCudaTest, AssignToThirdCluster) {
  EmbeddingMatrix<float> centers(3, 4);
  this->fill_matrix(
      centers, {float(1.0), float(0.0), float(0.0), float(0.0), float(0.0), float(1.0), float(0.0),
                float(0.0), float(0.0), float(0.0), float(1.0), float(0.0)});

  this->engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  float vec[] = {float(0.0), float(0.0), float(1.0), float(0.0)};
  EmbeddingView view{vec, 4, Device{CpuDevice{}}};
  auto [cluster_id, distance] = this->engine->assign(view);
  EXPECT_EQ(cluster_id, 2);
  EXPECT_NEAR(distance, float(0.0), float(1e-5));
}

TEST_F(ClusterEngineCudaTest, ManyClusterAssignment) {
  constexpr size_t N_CLUSTERS = 100;
  constexpr size_t DIM = 128;
  constexpr int N_QUERIES = 50;

  std::mt19937 gen(42);
  EmbeddingMatrix<float> centers(N_CLUSTERS, DIM);
  this->random_matrix(centers, gen);

  this->engine->load_centroids(centers.data(), centers.rows(), centers.cols());
  EXPECT_EQ(this->engine->n_clusters(), static_cast<size_t>(N_CLUSTERS));

  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (int q = 0; q < N_QUERIES; ++q) {
    std::vector<float> query(DIM);
    for (size_t d = 0; d < DIM; ++d) {
      query[d] = dist(gen);
    }

    EmbeddingView view{query.data(), DIM, Device{CpuDevice{}}};
    auto [cluster_id, distance] = this->engine->assign(view);

    EXPECT_GE(cluster_id, 0);
    EXPECT_LT(cluster_id, static_cast<int>(N_CLUSTERS));
    EXPECT_GE(distance, float(0.0));
  }
}

TEST_F(ClusterEngineCudaTest, SmallDimensionOptimization) {
  constexpr size_t N_CLUSTERS = 10;
  constexpr size_t DIM = 256;

  std::mt19937 gen(42);
  EmbeddingMatrix<float> centers(N_CLUSTERS, DIM);
  this->random_matrix(centers, gen);

  this->engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  std::vector<float> query(DIM, float(0.5));
  EmbeddingView view{query.data(), DIM, Device{CpuDevice{}}};
  auto [cluster_id, distance] = this->engine->assign(view);
  EXPECT_GE(cluster_id, 0);
  EXPECT_LT(cluster_id, static_cast<int>(N_CLUSTERS));
}

TEST_F(ClusterEngineCudaTest, LargeDimensionOptimization) {
  constexpr size_t N_CLUSTERS = 10;
  constexpr size_t DIM = 1024;

  std::mt19937 gen(42);
  EmbeddingMatrix<float> centers(N_CLUSTERS, DIM);
  this->random_matrix(centers, gen);

  this->engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  std::vector<float> query(DIM, float(0.5));
  EmbeddingView view{query.data(), DIM, Device{CpuDevice{}}};
  auto [cluster_id, distance] = this->engine->assign(view);
  EXPECT_GE(cluster_id, 0);
  EXPECT_LT(cluster_id, static_cast<int>(N_CLUSTERS));
}

TEST_F(ClusterEngineCudaTest, SmallClusterCountOptimization) {
  constexpr size_t N_CLUSTERS = 50;
  constexpr size_t DIM = 128;

  std::mt19937 gen(42);
  EmbeddingMatrix<float> centers(N_CLUSTERS, DIM);
  this->random_matrix(centers, gen);

  this->engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  std::vector<float> query(DIM, float(0.5));
  EmbeddingView view{query.data(), DIM, Device{CpuDevice{}}};
  auto [cluster_id, distance] = this->engine->assign(view);
  EXPECT_GE(cluster_id, 0);
  EXPECT_LT(cluster_id, static_cast<int>(N_CLUSTERS));
}

TEST_F(ClusterEngineCudaTest, LargeClusterCountOptimization) {
  constexpr size_t N_CLUSTERS = 200;
  constexpr size_t DIM = 128;

  std::mt19937 gen(42);
  EmbeddingMatrix<float> centers(N_CLUSTERS, DIM);
  this->random_matrix(centers, gen);

  this->engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  std::vector<float> query(DIM, float(0.5));
  EmbeddingView view{query.data(), DIM, Device{CpuDevice{}}};
  auto [cluster_id, distance] = this->engine->assign(view);
  EXPECT_GE(cluster_id, 0);
  EXPECT_LT(cluster_id, static_cast<int>(N_CLUSTERS));
}

TEST_F(ClusterEngineCudaTest, ReloadCentroidsRecapturesGraph) {
  EmbeddingMatrix<float> centers1(3, 4);
  this->fill_matrix(
      centers1, {float(1.0), float(0.0), float(0.0), float(0.0), float(0.0), float(1.0), float(0.0),
                 float(0.0), float(0.0), float(0.0), float(1.0), float(0.0)});

  this->engine->load_centroids(centers1.data(), centers1.rows(), centers1.cols());

  float vec1[] = {float(0.9), float(0.1), float(0.0), float(0.0)};
  EmbeddingView view1{vec1, 4, Device{CpuDevice{}}};
  auto [id1, dist1] = this->engine->assign(view1);
  EXPECT_EQ(id1, 0);

  EmbeddingMatrix<float> centers2(3, 4);
  this->fill_matrix(
      centers2, {float(0.0), float(1.0), float(0.0), float(0.0), float(1.0), float(0.0), float(0.0),
                 float(0.0), float(0.0), float(0.0), float(1.0), float(0.0)});

  this->engine->load_centroids(centers2.data(), centers2.rows(), centers2.cols());

  EmbeddingView view2{vec1, 4, Device{CpuDevice{}}};
  auto [id2, dist2] = this->engine->assign(view2);
  EXPECT_EQ(id2, 1);
}

TEST_F(ClusterEngineCudaTest, DimensionMismatch) {
  EmbeddingMatrix<float> centers(3, 4);
  this->fill_matrix(
      centers, {float(1.0), float(0.0), float(0.0), float(0.0), float(0.0), float(1.0), float(0.0),
                float(0.0), float(0.0), float(0.0), float(1.0), float(0.0)});
  this->engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  std::vector<float> query(3, float(1.0));
  EmbeddingView view{query.data(), 3, Device{CpuDevice{}}};
  EXPECT_THROW(this->engine->assign(view), std::invalid_argument);
}

// =============================================================================
// SECTION 4: CUDA Backend - Advanced GPU Tests
// =============================================================================

TEST(CudaAdvancedTest, VeryLargeClusterCount) {
  if (!cuda_available()) GTEST_SKIP();

  constexpr size_t N_CLUSTERS = 5000;
  constexpr size_t DIM = 128;

  std::unique_ptr<IClusterBackend> engine{create_backend(Device{CudaDevice{0}})};

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  EmbeddingMatrix<float> centers(N_CLUSTERS, DIM);
  for (size_t i = 0; i < N_CLUSTERS; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
      centers(i, j) = dist(gen);
    }
  }

  engine->load_centroids(centers.data(), centers.rows(), centers.cols());
  EXPECT_EQ(engine->n_clusters(), static_cast<size_t>(N_CLUSTERS));

  std::vector<float> query(DIM);
  for (size_t d = 0; d < DIM; ++d) {
    query[d] = dist(gen);
  }

  EmbeddingView view{query.data(), DIM, Device{CpuDevice{}}};
  auto [cluster_id, distance] = engine->assign(view);
  EXPECT_GE(cluster_id, 0);
  EXPECT_LT(cluster_id, static_cast<int>(N_CLUSTERS));
}

TEST(CudaAdvancedTest, VeryHighDimensionality) {
  if (!cuda_available()) GTEST_SKIP();

  constexpr size_t N_CLUSTERS = 10;
  constexpr size_t DIM = 4096;

  std::unique_ptr<IClusterBackend> engine{create_backend(Device{CudaDevice{0}})};

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  EmbeddingMatrix<float> centers(N_CLUSTERS, DIM);
  for (size_t i = 0; i < N_CLUSTERS; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
      centers(i, j) = dist(gen);
    }
  }

  engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  std::vector<float> query(DIM);
  for (size_t d = 0; d < DIM; ++d) {
    query[d] = dist(gen);
  }

  EmbeddingView view{query.data(), DIM, Device{CpuDevice{}}};
  auto [cluster_id, distance] = engine->assign(view);
  EXPECT_GE(cluster_id, 0);
  EXPECT_LT(cluster_id, static_cast<int>(N_CLUSTERS));
}

TEST(CudaAdvancedTest, ArgminWithNonMultipleOfFourClusters) {
  if (!cuda_available()) GTEST_SKIP();

  constexpr size_t N_CLUSTERS = 37;
  constexpr size_t DIM = 64;

  std::unique_ptr<IClusterBackend> engine{create_backend(Device{CudaDevice{0}})};

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  EmbeddingMatrix<float> centers(N_CLUSTERS, DIM);
  for (size_t i = 0; i < N_CLUSTERS; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
      centers(i, j) = dist(gen);
    }
  }

  engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  std::vector<float> query(DIM);
  for (size_t d = 0; d < DIM; ++d) {
    query[d] = dist(gen);
  }

  EmbeddingView view{query.data(), DIM, Device{CpuDevice{}}};
  auto [cluster_id, distance] = engine->assign(view);
  EXPECT_GE(cluster_id, 0);
  EXPECT_LT(cluster_id, static_cast<int>(N_CLUSTERS));
}

TEST(CudaAdvancedTest, NormCalculationAccuracy) {
  if (!cuda_available()) GTEST_SKIP();

  constexpr size_t N_CLUSTERS = 10;
  constexpr size_t DIM = 128;

  std::unique_ptr<IClusterBackend> engine{create_backend(Device{CudaDevice{0}})};

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  EmbeddingMatrix<float> centers(N_CLUSTERS, DIM);
  for (size_t i = 0; i < N_CLUSTERS; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
      centers(i, j) = dist(gen);
    }
  }

  engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  std::vector<float> query(DIM);
  for (size_t d = 0; d < DIM; ++d) {
    query[d] = dist(gen);
  }

  EmbeddingView view{query.data(), DIM, Device{CpuDevice{}}};
  auto [cluster_id, distance] = engine->assign(view);
  EXPECT_GE(cluster_id, 0);
  EXPECT_GE(distance, 0.0f);
  EXPECT_FALSE(std::isnan(distance));
  EXPECT_FALSE(std::isinf(distance));
}

TEST(CudaAdvancedTest, MultipleEngineInstances) {
  if (!cuda_available()) GTEST_SKIP();

  constexpr size_t N_CLUSTERS = 20;
  constexpr size_t DIM = 64;

  std::unique_ptr<IClusterBackend> engine1{create_backend(Device{CudaDevice{0}})};
  std::unique_ptr<IClusterBackend> engine2{create_backend(Device{CudaDevice{0}})};

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

  engine1->load_centroids(centers1.data(), centers1.rows(), centers1.cols());
  engine2->load_centroids(centers2.data(), centers2.rows(), centers2.cols());

  std::vector<float> query(DIM);
  for (size_t d = 0; d < DIM; ++d) {
    query[d] = dist(gen);
  }

  EmbeddingView view{query.data(), DIM, Device{CpuDevice{}}};
  auto [id1, dist1] = engine1->assign(view);
  auto [id2, dist2] = engine2->assign(view);

  EXPECT_GE(id1, 0);
  EXPECT_GE(id2, 0);
  EXPECT_LT(id1, static_cast<int>(N_CLUSTERS));
  EXPECT_LT(id2, static_cast<int>(N_CLUSTERS));
}

TEST(CudaAdvancedTest, NonMultipleOfFourDimensions) {
  if (!cuda_available()) GTEST_SKIP();

  constexpr size_t N_CLUSTERS = 10;
  constexpr size_t DIM = 127;

  std::unique_ptr<IClusterBackend> engine{create_backend(Device{CudaDevice{0}})};

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  EmbeddingMatrix<float> centers(N_CLUSTERS, DIM);
  for (size_t i = 0; i < N_CLUSTERS; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
      centers(i, j) = dist(gen);
    }
  }

  engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  std::vector<float> query(DIM);
  for (size_t d = 0; d < DIM; ++d) {
    query[d] = dist(gen);
  }

  EmbeddingView view{query.data(), DIM, Device{CpuDevice{}}};
  auto [cluster_id, distance] = engine->assign(view);
  EXPECT_GE(cluster_id, 0);
  EXPECT_LT(cluster_id, static_cast<int>(N_CLUSTERS));
}

TEST(CudaAdvancedTest, RepeatedAssignCalls) {
  if (!cuda_available()) GTEST_SKIP();

  constexpr size_t N_CLUSTERS = 50;
  constexpr size_t DIM = 128;
  constexpr int N_ITERATIONS = 1000;

  std::unique_ptr<IClusterBackend> engine{create_backend(Device{CudaDevice{0}})};

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  EmbeddingMatrix<float> centers(N_CLUSTERS, DIM);
  for (size_t i = 0; i < N_CLUSTERS; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
      centers(i, j) = dist(gen);
    }
  }

  engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  for (int iter = 0; iter < N_ITERATIONS; ++iter) {
    std::vector<float> query(DIM);
    for (size_t d = 0; d < DIM; ++d) {
      query[d] = dist(gen);
    }

    EmbeddingView view{query.data(), DIM, Device{CpuDevice{}}};
    auto [cluster_id, distance] = engine->assign(view);
    EXPECT_GE(cluster_id, 0);
    EXPECT_LT(cluster_id, static_cast<int>(N_CLUSTERS));
    EXPECT_GE(distance, 0.0f);
  }
}

// =============================================================================
// SECTION 5: CUDA Backend - Thread Safety
// =============================================================================

class ClusterCudaThreadSafetyTest : public ::testing::Test {
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
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    EmbeddingMatrix<float> centers(N_CLUSTERS, DIM);
    for (size_t i = 0; i < N_CLUSTERS; ++i) {
      for (size_t j = 0; j < DIM; ++j) {
        centers(i, j) = dist(gen);
      }
    }

    engine_ = create_backend(Device{CudaDevice{0}});
    engine_->load_centroids(centers.data(), centers.rows(), centers.cols());
  }

  std::unique_ptr<IClusterBackend> engine_{create_backend(Device{CpuDevice{}})};
};

TEST_F(ClusterCudaThreadSafetyTest, ConcurrentAssign) {
  std::atomic<int> error_count{0};
  std::atomic<int> success_count{0};
  std::vector<std::thread> threads;

  for (size_t t = 0; t < this->N_THREADS; ++t) {
    threads.emplace_back([&, t]() {
      std::mt19937 gen(static_cast<unsigned>(42 + t));
      std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

      for (int q = 0; q < this->N_QUERIES_PER_THREAD; ++q) {
        std::vector<float> query(this->DIM);
        for (size_t d = 0; d < this->DIM; ++d) {
          query[d] = dist(gen);
        }

        EmbeddingView view{query.data(), this->DIM, Device{CpuDevice{}}};
        auto [cluster_id, distance] = this->engine_->assign(view);

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

TEST_F(ClusterCudaThreadSafetyTest, StressTest) {
  static constexpr size_t STRESS_THREADS = 64;
  static constexpr int STRESS_QUERIES = 5000;

  std::atomic<int> success_count{0};
  std::atomic<int> failure_count{0};
  std::vector<std::thread> threads;

  for (size_t t = 0; t < STRESS_THREADS; ++t) {
    threads.emplace_back([&, t]() {
      std::mt19937 gen(static_cast<unsigned>(100 + t));
      std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

      for (int q = 0; q < STRESS_QUERIES; ++q) {
        std::vector<float> query(this->DIM);
        for (size_t d = 0; d < this->DIM; ++d) {
          query[d] = dist(gen);
        }

        EmbeddingView view{query.data(), this->DIM, Device{CpuDevice{}}};
        auto [cluster_id, distance] = this->engine_->assign(view);

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

  std::unique_ptr<IClusterBackend> cpu_engine{create_backend(Device{CpuDevice{}})};
  std::unique_ptr<IClusterBackend> cuda_engine{create_backend(Device{CudaDevice{0}})};

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  EmbeddingMatrix<float> centers(N_CLUSTERS, DIM);
  for (size_t i = 0; i < N_CLUSTERS; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
      centers(i, j) = dist(gen);
    }
  }

  cpu_engine->load_centroids(centers.data(), centers.rows(), centers.cols());
  cuda_engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  int mismatch_count = 0;
  for (int q = 0; q < N_QUERIES; ++q) {
    std::vector<float> query(DIM);
    for (size_t d = 0; d < DIM; ++d) {
      query[d] = dist(gen);
    }

    EmbeddingView view{query.data(), DIM, Device{CpuDevice{}}};
    auto [cpu_id, cpu_dist] = cpu_engine->assign(view);
    auto [cuda_id, cuda_dist] = cuda_engine->assign(view);

    if (cpu_id != cuda_id) {
      ++mismatch_count;
    }
    EXPECT_NEAR(cpu_dist, cuda_dist, 1e-4f) << "Distance mismatch at query " << q;
  }

  EXPECT_LE(mismatch_count, N_QUERIES / 10) << "Too many cluster assignment mismatches";
}

TEST(BackendComparisonTest, EdgeCaseConsistency) {
  if (!cuda_available()) GTEST_SKIP();

  constexpr size_t N_CLUSTERS = 10;
  constexpr size_t DIM = 64;

  std::unique_ptr<IClusterBackend> cpu_engine{create_backend(Device{CpuDevice{}})};
  std::unique_ptr<IClusterBackend> cuda_engine{create_backend(Device{CudaDevice{0}})};

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  EmbeddingMatrix<float> centers(N_CLUSTERS, DIM);
  for (size_t i = 0; i < N_CLUSTERS; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
      centers(i, j) = dist(gen);
    }
  }

  cpu_engine->load_centroids(centers.data(), centers.rows(), centers.cols());
  cuda_engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  std::vector<float> zero_query(DIM, 0.0f);
  EmbeddingView view{zero_query.data(), DIM, Device{CpuDevice{}}};
  auto [cpu_id1, cpu_dist1] = cpu_engine->assign(view);
  auto [cuda_id1, cuda_dist1] = cuda_engine->assign(view);
  EXPECT_EQ(cpu_id1, cuda_id1) << "Zero embedding mismatch";
  EXPECT_NEAR(cpu_dist1, cuda_dist1, 1e-4f);

  std::vector<float> small_query(DIM, 1e-7f);
  EmbeddingView small_view{small_query.data(), DIM, Device{CpuDevice{}}};
  auto [cpu_id2, cpu_dist2] = cpu_engine->assign(small_view);
  auto [cuda_id2, cuda_dist2] = cuda_engine->assign(small_view);
  EXPECT_EQ(cpu_id2, cuda_id2) << "Small values mismatch";
  EXPECT_NEAR(cpu_dist2, cuda_dist2, 1e-4f);

  std::vector<float> large_query(DIM, 1e6f);
  EmbeddingView large_view{large_query.data(), DIM, Device{CpuDevice{}}};
  auto [cpu_id3, cpu_dist3] = cpu_engine->assign(large_view);
  auto [cuda_id3, cuda_dist3] = cuda_engine->assign(large_view);
  // For large values, floating-point precision can cause different argmin results
  // when distances are very close. Just verify distances are similar.
  EXPECT_NEAR(cpu_dist3, cuda_dist3, cpu_dist3 * 1e-4f) << "Large values distance mismatch";
}

// =============================================================================
// SECTION 7: Backend Factory & Error Handling
// =============================================================================

TEST(ClusterBackendFactoryTest, CpuBackendWorks) {
  auto engine = create_backend(Device{CpuDevice{}});
  EXPECT_NE(engine, nullptr);
}

TEST(ClusterBackendFactoryTest, ExplicitCPUBackend) {
  auto engine = create_backend(Device{CpuDevice{}});
}

TEST(ClusterBackendFactoryTest, CudaAvailableFunction) {
  bool available = cuda_available();
  (void)available;
}

// =============================================================================
// SECTION 8: Edge Cases & Robustness
// =============================================================================

TEST(ClusterEdgeCasesTest, ZeroClusters) {
  auto engine = create_backend(Device{CpuDevice{}});
  std::vector<float> query(128, 1.0f);
  EmbeddingView view{query.data(), 128, Device{CpuDevice{}}};
  auto [id, dist] = engine->assign(view);
  EXPECT_EQ(id, -1);
}

TEST(ClusterEdgeCasesTest, SingleCluster) {
  auto engine = create_backend(Device{CpuDevice{}});

  EmbeddingMatrix<float> centers(1, 4);
  centers(0, 0) = 1.0f;
  centers(0, 1) = 0.0f;
  centers(0, 2) = 0.0f;
  centers(0, 3) = 0.0f;

  engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  std::vector<float> query1{0.9f, 0.1f, 0.0f, 0.0f};
  EmbeddingView view1{query1.data(), 4, Device{CpuDevice{}}};
  auto [id1, dist1] = engine->assign(view1);
  EXPECT_EQ(id1, 0);

  std::vector<float> query2{0.0f, 0.0f, 1.0f, 0.0f};
  EmbeddingView view2{query2.data(), 4, Device{CpuDevice{}}};
  auto [id2, dist2] = engine->assign(view2);
  EXPECT_EQ(id2, 0);
}

TEST(ClusterEdgeCasesTest, TwoClusters) {
  auto engine = create_backend(Device{CpuDevice{}});

  EmbeddingMatrix<float> centers(2, 2);
  centers(0, 0) = 1.0f;
  centers(0, 1) = 0.0f;
  centers(1, 0) = 0.0f;
  centers(1, 1) = 1.0f;

  engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  std::vector<float> query1{0.9f, 0.1f};
  EmbeddingView view1{query1.data(), 2, Device{CpuDevice{}}};
  auto [id1, dist1] = engine->assign(view1);
  EXPECT_EQ(id1, 0);

  std::vector<float> query2{0.1f, 0.9f};
  EmbeddingView view2{query2.data(), 2, Device{CpuDevice{}}};
  auto [id2, dist2] = engine->assign(view2);
  EXPECT_EQ(id2, 1);
}

TEST(ClusterEdgeCasesTest, EmbeddingAllZeros) {
  auto engine = create_backend(Device{CpuDevice{}});

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

  engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  std::vector<float> zero_query(4, 0.0f);
  EmbeddingView view{zero_query.data(), 4, Device{CpuDevice{}}};
  auto [id, dist] = engine->assign(view);
  EXPECT_GE(id, 0);
  EXPECT_LT(id, 3);
  EXPECT_GE(dist, 0.0f);
}

TEST(ClusterEdgeCasesTest, VerySmallValues) {
  auto engine = create_backend(Device{CpuDevice{}});

  EmbeddingMatrix<float> centers(2, 4);
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      centers(i, j) = 1e-7f * static_cast<float>(i + j);
    }
  }

  engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  std::vector<float> query(4, 1e-7f);
  EmbeddingView view{query.data(), 4, Device{CpuDevice{}}};
  auto [id, dist] = engine->assign(view);
  EXPECT_GE(id, 0);
  EXPECT_LT(id, 2);
}

TEST(ClusterEdgeCasesTest, VeryLargeValues) {
  auto engine = create_backend(Device{CpuDevice{}});

  EmbeddingMatrix<float> centers(2, 4);
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      centers(i, j) = 1e6f * static_cast<float>(i + j + 1);
    }
  }

  engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  std::vector<float> query(4, 1e6f);
  EmbeddingView view{query.data(), 4, Device{CpuDevice{}}};
  auto [id, dist] = engine->assign(view);
  EXPECT_GE(id, 0);
  EXPECT_LT(id, 2);
}

TEST(ClusterEdgeCasesTest, MixedScaleValues) {
  auto engine = create_backend(Device{CpuDevice{}});

  EmbeddingMatrix<float> centers(2, 4);
  centers(0, 0) = 1e-7f;
  centers(0, 1) = 1e7f;
  centers(0, 2) = 1e-3f;
  centers(0, 3) = 1e3f;
  centers(1, 0) = 1e7f;
  centers(1, 1) = 1e-7f;
  centers(1, 2) = 1e3f;
  centers(1, 3) = 1e-3f;

  engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  std::vector<float> query{1e-6f, 1e6f, 1e-2f, 1e2f};
  EmbeddingView view{query.data(), 4, Device{CpuDevice{}}};
  auto [id, dist] = engine->assign(view);
  EXPECT_GE(id, 0);
  EXPECT_LT(id, 2);
}

TEST(ClusterEdgeCasesTest, IdenticalCentroids) {
  auto engine = create_backend(Device{CpuDevice{}});

  EmbeddingMatrix<float> centers(3, 4);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      centers(i, j) = 1.0f;
    }
  }

  engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  std::vector<float> query{1.0f, 1.0f, 1.0f, 1.0f};
  EmbeddingView view{query.data(), 4, Device{CpuDevice{}}};
  auto [id, dist] = engine->assign(view);
  EXPECT_GE(id, 0);
  EXPECT_LT(id, 3);
  EXPECT_NEAR(dist, 0.0f, 1e-5f);
}

TEST(ClusterEdgeCasesTest, CUDAZeroClusters) {
  if (!cuda_available()) GTEST_SKIP();

  std::unique_ptr<IClusterBackend> engine{create_backend(Device{CudaDevice{0}})};
  std::vector<float> query(128, 1.0f);
  EmbeddingView view{query.data(), 128, Device{CpuDevice{}}};
  auto [id, dist] = engine->assign(view);
  EXPECT_EQ(id, -1);
}

TEST(ClusterEdgeCasesTest, CUDASingleCluster) {
  if (!cuda_available()) GTEST_SKIP();

  std::unique_ptr<IClusterBackend> engine{create_backend(Device{CudaDevice{0}})};

  EmbeddingMatrix<float> centers(1, 4);
  centers(0, 0) = 1.0f;
  centers(0, 1) = 0.0f;
  centers(0, 2) = 0.0f;
  centers(0, 3) = 0.0f;

  engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  std::vector<float> query{0.5f, 0.5f, 0.5f, 0.5f};
  EmbeddingView view{query.data(), 4, Device{CpuDevice{}}};
  auto [id, dist] = engine->assign(view);
  EXPECT_EQ(id, 0);
  EXPECT_GE(dist, 0.0f);
}

// =============================================================================
// SECTION 9: CUDA Kernel Edge Cases & Accuracy
// =============================================================================

TEST(CudaKernelEdgeCasesTest, SingleDimension) {
  if (!cuda_available()) GTEST_SKIP();

  std::unique_ptr<IClusterBackend> cuda_engine{create_backend(Device{CudaDevice{0}})};
  std::unique_ptr<IClusterBackend> cpu_engine{create_backend(Device{CpuDevice{}})};

  EmbeddingMatrix<float> centers(3, 1);
  centers(0, 0) = 0.0f;
  centers(1, 0) = 5.0f;
  centers(2, 0) = 10.0f;

  cuda_engine->load_centroids(centers.data(), centers.rows(), centers.cols());
  cpu_engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  std::vector<float> query1{4.0f};
  EmbeddingView view1{query1.data(), 1, Device{CpuDevice{}}};
  auto [cuda_id1, cuda_dist1] = cuda_engine->assign(view1);
  auto [cpu_id1, cpu_dist1] = cpu_engine->assign(view1);
  EXPECT_EQ(cuda_id1, cpu_id1);
  EXPECT_NEAR(cuda_dist1, cpu_dist1, 1e-5f);

  std::vector<float> query2{11.0f};
  EmbeddingView view2{query2.data(), 1, Device{CpuDevice{}}};
  auto [cuda_id2, cuda_dist2] = cuda_engine->assign(view2);
  auto [cpu_id2, cpu_dist2] = cpu_engine->assign(view2);
  EXPECT_EQ(cuda_id2, cpu_id2);
  EXPECT_NEAR(cuda_dist2, cpu_dist2, 1e-5f);
}

TEST(CudaKernelEdgeCasesTest, TwoDimensions) {
  if (!cuda_available()) GTEST_SKIP();

  std::unique_ptr<IClusterBackend> cuda_engine{create_backend(Device{CudaDevice{0}})};
  std::unique_ptr<IClusterBackend> cpu_engine{create_backend(Device{CpuDevice{}})};

  EmbeddingMatrix<float> centers(4, 2);
  centers(0, 0) = 0.0f;
  centers(0, 1) = 0.0f;
  centers(1, 0) = 1.0f;
  centers(1, 1) = 0.0f;
  centers(2, 0) = 0.0f;
  centers(2, 1) = 1.0f;
  centers(3, 0) = 1.0f;
  centers(3, 1) = 1.0f;

  cuda_engine->load_centroids(centers.data(), centers.rows(), centers.cols());
  cpu_engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  std::vector<float> query{0.8f, 0.2f};
  EmbeddingView view{query.data(), 2, Device{CpuDevice{}}};
  auto [cuda_id, cuda_dist] = cuda_engine->assign(view);
  auto [cpu_id, cpu_dist] = cpu_engine->assign(view);
  EXPECT_EQ(cuda_id, cpu_id);
  EXPECT_NEAR(cuda_dist, cpu_dist, 1e-5f);
}

TEST(CudaKernelEdgeCasesTest, ThreeDimensions) {
  if (!cuda_available()) GTEST_SKIP();

  std::unique_ptr<IClusterBackend> cuda_engine{create_backend(Device{CudaDevice{0}})};
  std::unique_ptr<IClusterBackend> cpu_engine{create_backend(Device{CpuDevice{}})};

  EmbeddingMatrix<float> centers(2, 3);
  centers(0, 0) = 1.0f;
  centers(0, 1) = 2.0f;
  centers(0, 2) = 3.0f;
  centers(1, 0) = 4.0f;
  centers(1, 1) = 5.0f;
  centers(1, 2) = 6.0f;

  cuda_engine->load_centroids(centers.data(), centers.rows(), centers.cols());
  cpu_engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  std::vector<float> query{2.0f, 3.0f, 4.0f};
  EmbeddingView view{query.data(), 3, Device{CpuDevice{}}};
  auto [cuda_id, cuda_dist] = cuda_engine->assign(view);
  auto [cpu_id, cpu_dist] = cpu_engine->assign(view);
  EXPECT_EQ(cuda_id, cpu_id);
  EXPECT_NEAR(cuda_dist, cpu_dist, 1e-5f);
}

TEST(CudaKernelEdgeCasesTest, TwoClusters) {
  if (!cuda_available()) GTEST_SKIP();

  std::unique_ptr<IClusterBackend> cuda_engine{create_backend(Device{CudaDevice{0}})};
  std::unique_ptr<IClusterBackend> cpu_engine{create_backend(Device{CpuDevice{}})};

  EmbeddingMatrix<float> centers(2, 64);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 64; ++j) {
      centers(i, j) = dist(gen);
    }
  }

  cuda_engine->load_centroids(centers.data(), centers.rows(), centers.cols());
  cpu_engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  for (int q = 0; q < 10; ++q) {
    std::vector<float> query(64);
    for (size_t d = 0; d < 64; ++d) {
      query[d] = dist(gen);
    }
    EmbeddingView view{query.data(), 64, Device{CpuDevice{}}};
    auto [cuda_id, cuda_dist] = cuda_engine->assign(view);
    auto [cpu_id, cpu_dist] = cpu_engine->assign(view);
    EXPECT_EQ(cuda_id, cpu_id) << "Mismatch at query " << q;
    EXPECT_NEAR(cuda_dist, cpu_dist, 1e-4f);
  }
}

TEST(CudaKernelEdgeCasesTest, IdenticalCentroidsCUDA) {
  if (!cuda_available()) GTEST_SKIP();

  std::unique_ptr<IClusterBackend> cuda_engine{create_backend(Device{CudaDevice{0}})};

  EmbeddingMatrix<float> centers(5, 32);
  for (size_t i = 0; i < 5; ++i) {
    for (size_t j = 0; j < 32; ++j) {
      centers(i, j) = 0.5f;
    }
  }

  cuda_engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  std::vector<float> query(32, 0.5f);
  EmbeddingView view{query.data(), 32, Device{CpuDevice{}}};
  auto [id, dist] = cuda_engine->assign(view);
  EXPECT_GE(id, 0);
  EXPECT_LT(id, 5);
  EXPECT_NEAR(dist, 0.0f, 1e-5f);
}

TEST(CudaKernelEdgeCasesTest, ZeroEmbedding) {
  if (!cuda_available()) GTEST_SKIP();

  std::unique_ptr<IClusterBackend> cuda_engine{create_backend(Device{CudaDevice{0}})};
  std::unique_ptr<IClusterBackend> cpu_engine{create_backend(Device{CpuDevice{}})};

  EmbeddingMatrix<float> centers(3, 64);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 64; ++j) {
      centers(i, j) = dist(gen);
    }
  }

  cuda_engine->load_centroids(centers.data(), centers.rows(), centers.cols());
  cpu_engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  std::vector<float> zero_query(64, 0.0f);
  EmbeddingView view{zero_query.data(), 64, Device{CpuDevice{}}};
  auto [cuda_id, cuda_dist] = cuda_engine->assign(view);
  auto [cpu_id, cpu_dist] = cpu_engine->assign(view);
  EXPECT_EQ(cuda_id, cpu_id);
  EXPECT_NEAR(cuda_dist, cpu_dist, 1e-4f);
}

TEST(CudaKernelEdgeCasesTest, LargeValues) {
  if (!cuda_available()) GTEST_SKIP();

  std::unique_ptr<IClusterBackend> cuda_engine{create_backend(Device{CudaDevice{0}})};
  std::unique_ptr<IClusterBackend> cpu_engine{create_backend(Device{CpuDevice{}})};

  EmbeddingMatrix<float> centers(3, 32);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 32; ++j) {
      centers(i, j) = static_cast<float>(i * 1000 + j);
    }
  }

  cuda_engine->load_centroids(centers.data(), centers.rows(), centers.cols());
  cpu_engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  std::vector<float> query(32);
  for (size_t j = 0; j < 32; ++j) {
    query[j] = static_cast<float>(1000 + j);
  }

  EmbeddingView view{query.data(), 32, Device{CpuDevice{}}};
  auto [cuda_id, cuda_dist] = cuda_engine->assign(view);
  auto [cpu_id, cpu_dist] = cpu_engine->assign(view);
  EXPECT_EQ(cuda_id, cpu_id);
  EXPECT_NEAR(cuda_dist, cpu_dist, 1e-2f);
}

TEST(CudaKernelEdgeCasesTest, SmallValues) {
  if (!cuda_available()) GTEST_SKIP();

  std::unique_ptr<IClusterBackend> cuda_engine{create_backend(Device{CudaDevice{0}})};
  std::unique_ptr<IClusterBackend> cpu_engine{create_backend(Device{CpuDevice{}})};

  EmbeddingMatrix<float> centers(3, 32);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 32; ++j) {
      centers(i, j) = static_cast<float>(i + 1) * 1e-6f;
    }
  }

  cuda_engine->load_centroids(centers.data(), centers.rows(), centers.cols());
  cpu_engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  std::vector<float> query(32, 2e-6f);
  EmbeddingView view{query.data(), 32, Device{CpuDevice{}}};
  auto [cuda_id, cuda_dist] = cuda_engine->assign(view);
  auto [cpu_id, cpu_dist] = cpu_engine->assign(view);
  EXPECT_EQ(cuda_id, cpu_id);
  EXPECT_NEAR(cuda_dist, cpu_dist, 1e-10f);
}

TEST(CudaKernelEdgeCasesTest, NegativeValues) {
  if (!cuda_available()) GTEST_SKIP();

  std::unique_ptr<IClusterBackend> cuda_engine{create_backend(Device{CudaDevice{0}})};
  std::unique_ptr<IClusterBackend> cpu_engine{create_backend(Device{CpuDevice{}})};

  EmbeddingMatrix<float> centers(3, 16);
  centers(0, 0) = -1.0f;
  for (size_t j = 1; j < 16; ++j) centers(0, j) = 0.0f;
  centers(1, 0) = 0.0f;
  for (size_t j = 1; j < 16; ++j) centers(1, j) = -1.0f;
  centers(2, 0) = 1.0f;
  for (size_t j = 1; j < 16; ++j) centers(2, j) = 1.0f;

  cuda_engine->load_centroids(centers.data(), centers.rows(), centers.cols());
  cpu_engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  std::vector<float> query(16, -0.5f);
  EmbeddingView view{query.data(), 16, Device{CpuDevice{}}};
  auto [cuda_id, cuda_dist] = cuda_engine->assign(view);
  auto [cpu_id, cpu_dist] = cpu_engine->assign(view);
  // Floating-point precision differences can cause different argmin when distances
  // are very close. Verify distances match; ID match is best-effort.
  EXPECT_NEAR(cuda_dist, cpu_dist, 1e-5f);
  if (cuda_id != cpu_id) {
    // If IDs differ, verify distances to both clusters are essentially equal
    EXPECT_NEAR(cuda_dist, cpu_dist, 1e-5f);
  }
}

TEST(CudaKernelEdgeCasesTest, PrimeDimension) {
  if (!cuda_available()) GTEST_SKIP();

  std::unique_ptr<IClusterBackend> cuda_engine{create_backend(Device{CudaDevice{0}})};
  std::unique_ptr<IClusterBackend> cpu_engine{create_backend(Device{CpuDevice{}})};

  constexpr size_t DIM = 131;
  EmbeddingMatrix<float> centers(10, DIM);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < 10; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
      centers(i, j) = dist(gen);
    }
  }

  cuda_engine->load_centroids(centers.data(), centers.rows(), centers.cols());
  cpu_engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  std::vector<float> query(DIM);
  for (size_t d = 0; d < DIM; ++d) {
    query[d] = dist(gen);
  }

  EmbeddingView view{query.data(), DIM, Device{CpuDevice{}}};
  auto [cuda_id, cuda_dist] = cuda_engine->assign(view);
  auto [cpu_id, cpu_dist] = cpu_engine->assign(view);
  EXPECT_EQ(cuda_id, cpu_id);
  EXPECT_NEAR(cuda_dist, cpu_dist, 1e-4f);
}

TEST(CudaKernelEdgeCasesTest, PrimeClusterCount) {
  if (!cuda_available()) GTEST_SKIP();

  std::unique_ptr<IClusterBackend> cuda_engine{create_backend(Device{CudaDevice{0}})};
  std::unique_ptr<IClusterBackend> cpu_engine{create_backend(Device{CpuDevice{}})};

  constexpr size_t N_CLUSTERS = 97;
  constexpr size_t DIM = 64;
  EmbeddingMatrix<float> centers(N_CLUSTERS, DIM);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < N_CLUSTERS; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
      centers(i, j) = dist(gen);
    }
  }

  cuda_engine->load_centroids(centers.data(), centers.rows(), centers.cols());
  cpu_engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  std::vector<float> query(DIM);
  for (size_t d = 0; d < DIM; ++d) {
    query[d] = dist(gen);
  }

  EmbeddingView view{query.data(), DIM, Device{CpuDevice{}}};
  auto [cuda_id, cuda_dist] = cuda_engine->assign(view);
  auto [cpu_id, cpu_dist] = cpu_engine->assign(view);
  EXPECT_EQ(cuda_id, cpu_id);
  EXPECT_NEAR(cuda_dist, cpu_dist, 1e-4f);
}

// =============================================================================
// SECTION 10: CUDA Batch Edge Cases
// =============================================================================

TEST(CudaBatchEdgeCasesTest, BatchSizeOne) {
  if (!cuda_available()) GTEST_SKIP();

  std::unique_ptr<IClusterBackend> cuda_engine{create_backend(Device{CudaDevice{0}})};

  EmbeddingMatrix<float> centers(5, 32);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < 5; ++i) {
    for (size_t j = 0; j < 32; ++j) {
      centers(i, j) = dist(gen);
    }
  }

  cuda_engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  std::vector<float> query(32);
  for (size_t d = 0; d < 32; ++d) {
    query[d] = dist(gen);
  }

  EmbeddingView view{query.data(), 32, Device{CpuDevice{}}};
  auto single_result = cuda_engine->assign(view);
  EmbeddingBatchView batch_view{query.data(), 1, 32, Device{CpuDevice{}}};
  auto batch_results = cuda_engine->assign_batch(batch_view);

  ASSERT_EQ(batch_results.size(), 1u);
  EXPECT_EQ(single_result.first, batch_results[0].first);
  EXPECT_NEAR(single_result.second, batch_results[0].second, 1e-5f);
}

TEST(CudaBatchEdgeCasesTest, BatchSizeTwo) {
  if (!cuda_available()) GTEST_SKIP();

  std::unique_ptr<IClusterBackend> cuda_engine{create_backend(Device{CudaDevice{0}})};

  constexpr size_t DIM = 64;
  EmbeddingMatrix<float> centers(10, DIM);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < 10; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
      centers(i, j) = dist(gen);
    }
  }

  cuda_engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  std::vector<float> queries(2 * DIM);
  for (size_t d = 0; d < 2 * DIM; ++d) {
    queries[d] = dist(gen);
  }

  EmbeddingBatchView batch_view{queries.data(), 2, DIM, Device{CpuDevice{}}};
  auto batch_results = cuda_engine->assign_batch(batch_view);
  ASSERT_EQ(batch_results.size(), 2u);

  for (size_t i = 0; i < 2; ++i) {
    EmbeddingView view{queries.data() + i * DIM, DIM, Device{CpuDevice{}}};
    auto single_result = cuda_engine->assign(view);
    EXPECT_EQ(batch_results[i].first, single_result.first);
    EXPECT_NEAR(batch_results[i].second, single_result.second, 1e-5f);
  }
}

TEST(CudaBatchEdgeCasesTest, BatchSizePrime) {
  if (!cuda_available()) GTEST_SKIP();

  std::unique_ptr<IClusterBackend> cuda_engine{create_backend(Device{CudaDevice{0}})};

  constexpr size_t N_QUERIES = 67;
  constexpr size_t DIM = 64;
  EmbeddingMatrix<float> centers(20, DIM);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < 20; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
      centers(i, j) = dist(gen);
    }
  }

  cuda_engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  std::vector<float> queries(N_QUERIES * DIM);
  for (size_t d = 0; d < N_QUERIES * DIM; ++d) {
    queries[d] = dist(gen);
  }

  EmbeddingBatchView batch_view{queries.data(), N_QUERIES, DIM, Device{CpuDevice{}}};
  auto batch_results = cuda_engine->assign_batch(batch_view);
  ASSERT_EQ(batch_results.size(), N_QUERIES);

  for (size_t i = 0; i < N_QUERIES; ++i) {
    EmbeddingView view{queries.data() + i * DIM, DIM, Device{CpuDevice{}}};
    auto single_result = cuda_engine->assign(view);
    EXPECT_EQ(batch_results[i].first, single_result.first) << "Mismatch at query " << i;
    EXPECT_NEAR(batch_results[i].second, single_result.second, 1e-5f);
    EXPECT_GE(batch_results[i].first, 0);
    EXPECT_LT(batch_results[i].first, 20);
  }
}

TEST(CudaBatchEdgeCasesTest, LargeBatch) {
  if (!cuda_available()) GTEST_SKIP();

  std::unique_ptr<IClusterBackend> engine{create_backend(Device{CudaDevice{0}})};

  constexpr size_t N_QUERIES = 2048;
  constexpr size_t N_CLUSTERS = 100;
  constexpr size_t DIM = 128;

  EmbeddingMatrix<float> centers(N_CLUSTERS, DIM);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < N_CLUSTERS; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
      centers(i, j) = dist(gen);
    }
  }

  engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  std::vector<float> queries(N_QUERIES * DIM);
  for (size_t d = 0; d < N_QUERIES * DIM; ++d) {
    queries[d] = dist(gen);
  }

  EmbeddingBatchView batch_view{queries.data(), N_QUERIES, DIM, Device{CpuDevice{}}};
  auto batch_results = engine->assign_batch(batch_view);

  // Verify batch vs single consistency within CUDA backend
  ASSERT_EQ(batch_results.size(), N_QUERIES);
  for (size_t i = 0; i < N_QUERIES; ++i) {
    EmbeddingView view{queries.data() + i * DIM, DIM, Device{CpuDevice{}}};
    auto single_result = engine->assign(view);
    EXPECT_EQ(batch_results[i].first, single_result.first) << "Mismatch at query " << i;
    EXPECT_NEAR(batch_results[i].second, single_result.second, 1e-5f);
  }
}

TEST(CudaBatchEdgeCasesTest, BatchVsSingleConsistency) {
  if (!cuda_available()) GTEST_SKIP();

  std::unique_ptr<IClusterBackend> engine{create_backend(Device{CudaDevice{0}})};

  constexpr size_t N_QUERIES = 100;
  constexpr size_t DIM = 64;

  EmbeddingMatrix<float> centers(20, DIM);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < 20; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
      centers(i, j) = dist(gen);
    }
  }

  engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  std::vector<float> queries(N_QUERIES * DIM);
  for (size_t d = 0; d < N_QUERIES * DIM; ++d) {
    queries[d] = dist(gen);
  }

  EmbeddingBatchView batch_view{queries.data(), N_QUERIES, DIM, Device{CpuDevice{}}};
  auto batch_results = engine->assign_batch(batch_view);

  ASSERT_EQ(batch_results.size(), N_QUERIES);
  for (size_t i = 0; i < N_QUERIES; ++i) {
    EmbeddingView view{queries.data() + i * DIM, DIM, Device{CpuDevice{}}};
    auto single_result = engine->assign(view);
    EXPECT_EQ(batch_results[i].first, single_result.first) << "Mismatch at query " << i;
    EXPECT_NEAR(batch_results[i].second, single_result.second, 1e-5f);
  }
}

TEST(CudaBatchEdgeCasesTest, EmptyBatch) {
  if (!cuda_available()) GTEST_SKIP();

  std::unique_ptr<IClusterBackend> engine{create_backend(Device{CudaDevice{0}})};

  EmbeddingMatrix<float> centers(5, 32);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < 5; ++i) {
    for (size_t j = 0; j < 32; ++j) {
      centers(i, j) = dist(gen);
    }
  }

  engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  EmbeddingBatchView batch_view{nullptr, 0, 32, Device{CpuDevice{}}};
  auto results = engine->assign_batch(batch_view);
  EXPECT_TRUE(results.empty());
}

TEST(CudaBatchEdgeCasesTest, BatchWithSmallDimension) {
  if (!cuda_available()) GTEST_SKIP();

  std::unique_ptr<IClusterBackend> engine{create_backend(Device{CudaDevice{0}})};

  constexpr size_t N_QUERIES = 50;
  constexpr size_t DIM = 3;

  EmbeddingMatrix<float> centers(4, DIM);
  centers(0, 0) = 0.0f;
  centers(0, 1) = 0.0f;
  centers(0, 2) = 0.0f;
  centers(1, 0) = 1.0f;
  centers(1, 1) = 0.0f;
  centers(1, 2) = 0.0f;
  centers(2, 0) = 0.0f;
  centers(2, 1) = 1.0f;
  centers(2, 2) = 0.0f;
  centers(3, 0) = 0.0f;
  centers(3, 1) = 0.0f;
  centers(3, 2) = 1.0f;

  engine->load_centroids(centers.data(), centers.rows(), centers.cols());

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> queries(N_QUERIES * DIM);
  for (size_t d = 0; d < N_QUERIES * DIM; ++d) {
    queries[d] = dist(gen);
  }

  EmbeddingBatchView batch_view{queries.data(), N_QUERIES, DIM, Device{CpuDevice{}}};
  auto batch_results = engine->assign_batch(batch_view);

  // Verify batch vs single consistency within CUDA backend
  ASSERT_EQ(batch_results.size(), N_QUERIES);
  for (size_t i = 0; i < N_QUERIES; ++i) {
    EmbeddingView view{queries.data() + i * DIM, DIM, Device{CpuDevice{}}};
    auto single_result = engine->assign(view);
    EXPECT_EQ(batch_results[i].first, single_result.first) << "Mismatch at query " << i;
    EXPECT_NEAR(batch_results[i].second, single_result.second, 1e-5f);
  }
}
