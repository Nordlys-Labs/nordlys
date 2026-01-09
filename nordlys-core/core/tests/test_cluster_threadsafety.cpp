#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <nordlys_core/cluster.hpp>
#include <thread>
#include <vector>

template <typename Scalar> class ClusterThreadSafetyTestT : public ::testing::Test {
protected:
  static constexpr int N_THREADS = 16;
  static constexpr int N_QUERIES_PER_THREAD = 1000;
  static constexpr int N_CLUSTERS = 100;
  static constexpr int DIM = 128;

  void SetUp() override {
    EmbeddingMatrixT<Scalar> centers = EmbeddingMatrixT<Scalar>::Random(N_CLUSTERS, DIM);

    engine_.load_centroids(centers);
  }

  ClusterEngineT<Scalar> engine_{ClusterBackendType::Cpu};
};

using ScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(ClusterThreadSafetyTestT, ScalarTypes);

TYPED_TEST(ClusterThreadSafetyTestT, ConcurrentAssign) {
  std::atomic<int> error_count{0};
  std::atomic<int> success_count{0};
  std::vector<std::thread> threads;

  for (int t = 0; t < this->N_THREADS; ++t) {
    threads.emplace_back([&]() {
      for (int q = 0; q < this->N_QUERIES_PER_THREAD; ++q) {
        EmbeddingVectorT<TypeParam> query = EmbeddingVectorT<TypeParam>::Random(this->DIM);

        auto [cluster_id, distance] = this->engine_.assign(query);

        if (cluster_id >= 0 && cluster_id < this->N_CLUSTERS && distance >= 0
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
  EXPECT_EQ(success_count.load(), this->N_THREADS * this->N_QUERIES_PER_THREAD);
}

TYPED_TEST(ClusterThreadSafetyTestT, StressTest) {
  static constexpr int STRESS_THREADS = 64;
  static constexpr int STRESS_QUERIES = 5000;

  std::atomic<int> success_count{0};
  std::atomic<int> failure_count{0};
  std::vector<std::thread> threads;

  for (int t = 0; t < STRESS_THREADS; ++t) {
    threads.emplace_back([&]() {
      for (int q = 0; q < STRESS_QUERIES; ++q) {
        EmbeddingVectorT<TypeParam> query = EmbeddingVectorT<TypeParam>::Random(this->DIM);

        auto [cluster_id, distance] = this->engine_.assign(query);

        if (cluster_id >= 0 && cluster_id < this->N_CLUSTERS && distance >= 0
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
  EXPECT_EQ(success_count.load(), STRESS_THREADS * STRESS_QUERIES);
}
