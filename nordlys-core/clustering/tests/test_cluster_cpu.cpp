#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <limits>
#include <nordlys/clustering/cluster.hpp>
#include <nordlys/common/device.hpp>
#include <nordlys/clustering/embedding_view.hpp>
#include <nordlys/common/matrix.hpp>
#include <random>
#include <thread>
#include <vector>

using namespace nordlys::clustering;

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
