#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <nordlys_core/cluster.hpp>
#include <random>
#include <thread>
#include <vector>

template <typename Scalar> class ClusterThreadSafetyTestT : public ::testing::Test {
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

using ScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(ClusterThreadSafetyTestT, ScalarTypes);

TYPED_TEST(ClusterThreadSafetyTestT, ConcurrentAssign) {
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

TYPED_TEST(ClusterThreadSafetyTestT, StressTest) {
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
