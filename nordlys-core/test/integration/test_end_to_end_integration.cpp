#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <nordlys/checkpoint/checkpoint.hpp>
#include <nordlys/common/device.hpp>
#include <nordlys/clustering/embedding_view.hpp>
#include <nordlys/routing/nordlys.hpp>
#include <vector>

#include "../fixtures/fixtures_loader.hpp"

namespace fs = std::filesystem;

class EndToEndIntegrationTest : public ::testing::Test {
protected:
  static fs::path get_test_data_dir() {
    return fs::path(__FILE__).parent_path().parent_path() / "fixtures";
  }

  static fs::path get_checkpoint_path(const std::string& name) {
    return get_test_data_dir() / name;
  }
};

TEST_F(EndToEndIntegrationTest, RouteLargeNumberOfEmbeddings) {
  auto checkpoint_path = get_checkpoint_path("valid_checkpoint_f32.json");
  auto checkpoint = NordlysCheckpoint::from_json(checkpoint_path.string());
  auto result = Nordlys::from_checkpoint(std::move(checkpoint));
  ASSERT_TRUE(result) << "Failed to create router: " << result.error();

  auto router = std::move(result.value());
  size_t embedding_dim = router.get_embedding_dim();

  auto embeddings = nordlys::testing::FixturesLoader::generate_embeddings(1000, embedding_dim);

  size_t successful_routes = 0;
  for (const auto& emb : embeddings) {
    EmbeddingView view{emb.data(), emb.size(), Device{CpuDevice{}}};
    auto response = router.route(view);
    EXPECT_FALSE(response.selected_model.empty());
    EXPECT_GE(response.cluster_id, 0);
    EXPECT_LT(response.cluster_id, static_cast<int>(router.get_n_clusters()));
    EXPECT_GE(response.cluster_distance, 0.0f);
    ++successful_routes;
  }

  EXPECT_EQ(successful_routes, 1000);
}

TEST_F(EndToEndIntegrationTest, RouteClusteredEmbeddings) {
  auto checkpoint_path = get_checkpoint_path("valid_checkpoint_f32.json");
  auto checkpoint = NordlysCheckpoint::from_json(checkpoint_path.string());
  auto result = Nordlys::from_checkpoint(std::move(checkpoint));
  ASSERT_TRUE(result);

  auto router = std::move(result.value());
  size_t embedding_dim = router.get_embedding_dim();
  size_t n_clusters = router.get_n_clusters();

  auto embeddings = nordlys::testing::FixturesLoader::generate_clustered_embeddings(
      1000, embedding_dim, n_clusters);

  std::vector<int> cluster_counts(n_clusters, 0);

  for (const auto& emb : embeddings) {
    EmbeddingView view{emb.data(), emb.size(), Device{CpuDevice{}}};
    auto response = router.route(view);
    EXPECT_GE(response.cluster_id, 0);
    EXPECT_LT(response.cluster_id, static_cast<int>(n_clusters));
    ++cluster_counts[response.cluster_id];
  }

  for (size_t i = 0; i < n_clusters; ++i) {
    EXPECT_GT(cluster_counts[i], 0) << "Cluster " << i << " had no assignments";
  }
}

TEST_F(EndToEndIntegrationTest, RoutingThroughput) {
  auto checkpoint_path = get_checkpoint_path("valid_checkpoint_f32.json");
  auto checkpoint = NordlysCheckpoint::from_json(checkpoint_path.string());
  auto result = Nordlys::from_checkpoint(std::move(checkpoint));
  ASSERT_TRUE(result);

  auto router = std::move(result.value());
  size_t embedding_dim = router.get_embedding_dim();

  auto embeddings = nordlys::testing::FixturesLoader::generate_embeddings(100, embedding_dim);

  std::unordered_map<std::string, int> model_counts;

  for (const auto& emb : embeddings) {
    EmbeddingView view{emb.data(), emb.size(), Device{CpuDevice{}}};
    auto response = router.route(view);
    ++model_counts[response.selected_model];
  }

  EXPECT_GT(model_counts.size(), 0);
}

TEST_F(EndToEndIntegrationTest, CheckpointRoundTripIntegration) {
  auto original_path = get_checkpoint_path("valid_checkpoint_f32.json");
  auto original_checkpoint = NordlysCheckpoint::from_json(original_path.string());

  auto result1 = Nordlys::from_checkpoint(NordlysCheckpoint(original_checkpoint));
  ASSERT_TRUE(result1);
  auto router1 = std::move(result1.value());

  auto msgpack_str = original_checkpoint.to_msgpack_string();
  auto reloaded_checkpoint = NordlysCheckpoint::from_msgpack_string(msgpack_str);

  auto result2 = Nordlys::from_checkpoint(std::move(reloaded_checkpoint));
  ASSERT_TRUE(result2);
  auto router2 = std::move(result2.value());

  EXPECT_EQ(router1.get_n_clusters(), router2.get_n_clusters());
  EXPECT_EQ(router1.get_embedding_dim(), router2.get_embedding_dim());
  EXPECT_EQ(router1.get_supported_models().size(), router2.get_supported_models().size());

  size_t embedding_dim = router1.get_embedding_dim();
  auto embeddings = nordlys::testing::FixturesLoader::generate_embeddings(100, embedding_dim);

  for (const auto& emb : embeddings) {
    EmbeddingView view{emb.data(), emb.size(), Device{CpuDevice{}}};
    auto resp1 = router1.route(view);
    auto resp2 = router2.route(view);

    EXPECT_EQ(resp1.cluster_id, resp2.cluster_id);
    EXPECT_NEAR(resp1.cluster_distance, resp2.cluster_distance, 1e-5f);
  }
}

TEST_F(EndToEndIntegrationTest, ClusterBoundaryBehavior) {
  auto checkpoint_path = get_checkpoint_path("valid_checkpoint_f32.json");
  auto checkpoint = NordlysCheckpoint::from_json(checkpoint_path.string());
  auto result = Nordlys::from_checkpoint(std::move(checkpoint));
  ASSERT_TRUE(result);

  auto router = std::move(result.value());
  size_t embedding_dim = router.get_embedding_dim();

  auto embeddings = nordlys::testing::FixturesLoader::generate_embeddings(200, embedding_dim);

  for (const auto& emb : embeddings) {
    EmbeddingView view{emb.data(), emb.size(), Device{CpuDevice{}}};
    auto response = router.route(view);

    EXPECT_GE(response.cluster_id, 0);
    EXPECT_LT(response.cluster_id, static_cast<int>(router.get_n_clusters()));

    EXPECT_GE(response.cluster_distance, 0.0f);
    EXPECT_LT(response.cluster_distance, 2.0f);
  }
}

TEST_F(EndToEndIntegrationTest, ModelFilteringThroughput) {
  auto checkpoint_path = get_checkpoint_path("valid_checkpoint_f32.json");
  auto checkpoint = NordlysCheckpoint::from_json(checkpoint_path.string());
  auto result = Nordlys::from_checkpoint(std::move(checkpoint));
  ASSERT_TRUE(result);

  auto router = std::move(result.value());
  size_t embedding_dim = router.get_embedding_dim();
  auto models = router.get_supported_models();
  ASSERT_GT(models.size(), 0);

  std::vector<std::string> filter = {models[0]};

  auto embeddings = nordlys::testing::FixturesLoader::generate_embeddings(100, embedding_dim);

  for (const auto& emb : embeddings) {
    EmbeddingView view{emb.data(), emb.size(), Device{CpuDevice{}}};
    auto response = router.route(view, filter);
    EXPECT_EQ(response.selected_model, models[0]);
  }
}

TEST_F(EndToEndIntegrationTest, AlternativesConsistency) {
  auto checkpoint_path = get_checkpoint_path("valid_checkpoint_f32.json");
  auto checkpoint = NordlysCheckpoint::from_json(checkpoint_path.string());
  auto result = Nordlys::from_checkpoint(std::move(checkpoint));
  ASSERT_TRUE(result);

  auto router = std::move(result.value());
  size_t embedding_dim = router.get_embedding_dim();
  auto models = router.get_supported_models();

  auto embeddings = nordlys::testing::FixturesLoader::generate_embeddings(50, embedding_dim);

  for (const auto& emb : embeddings) {
    EmbeddingView view{emb.data(), emb.size(), Device{CpuDevice{}}};
    auto response = router.route(view);

    EXPECT_FALSE(response.selected_model.empty());
    EXPECT_LE(response.alternatives.size(), models.size() - 1);

    for (const auto& alt : response.alternatives) {
      EXPECT_NE(alt, response.selected_model);

      EXPECT_TRUE(std::find(models.begin(), models.end(), alt) != models.end());
    }
  }
}

TEST_F(EndToEndIntegrationTest, ConsistentClusterAssignmentAcrossInvocations) {
  auto checkpoint_path = get_checkpoint_path("valid_checkpoint_f32.json");
  auto checkpoint = NordlysCheckpoint::from_json(checkpoint_path.string());
  auto result = Nordlys::from_checkpoint(std::move(checkpoint));
  ASSERT_TRUE(result);

  auto router = std::move(result.value());
  size_t embedding_dim = router.get_embedding_dim();

  auto embeddings = nordlys::testing::FixturesLoader::generate_embeddings(50, embedding_dim);

  for (const auto& emb : embeddings) {
    EmbeddingView view{emb.data(), emb.size(), Device{CpuDevice{}}};
    auto resp1 = router.route(view);
    auto resp2 = router.route(view);

    EXPECT_EQ(resp1.cluster_id, resp2.cluster_id);
    EXPECT_FLOAT_EQ(resp1.cluster_distance, resp2.cluster_distance);
  }
}

#ifdef NORDLYS_ENABLE_CUDA
TEST_F(EndToEndIntegrationTest, CUDABackendThroughputTest) {
  auto checkpoint_path = get_checkpoint_path("valid_checkpoint_f32.json");
  auto checkpoint = NordlysCheckpoint::from_json(checkpoint_path.string());
  auto result = Nordlys::from_checkpoint(std::move(checkpoint));
  ASSERT_TRUE(result);

  auto router = std::move(result.value());
  size_t embedding_dim = router.get_embedding_dim();

  auto embeddings = nordlys::testing::FixturesLoader::generate_embeddings(1000, embedding_dim);

  size_t successful_routes = 0;
  for (const auto& emb : embeddings) {
    EmbeddingView view{emb.data(), emb.size(), Device{CpuDevice{}}};
    auto response = router.route(view);
    EXPECT_FALSE(response.selected_model.empty());
    EXPECT_GE(response.cluster_id, 0);
    ++successful_routes;
  }

  EXPECT_EQ(successful_routes, 1000);
}
#endif
