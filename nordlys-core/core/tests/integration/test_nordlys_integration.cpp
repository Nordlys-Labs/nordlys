#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <nordlys_core/checkpoint.hpp>
#include <nordlys_core/device.hpp>
#include <nordlys_core/embedding_view.hpp>
#include <nordlys_core/nordlys.hpp>
#include <vector>

namespace fs = std::filesystem;

class NordlysIntegrationTest : public ::testing::Test {
protected:
  static fs::path get_test_data_dir() {
    return fs::path(__FILE__).parent_path().parent_path() / "fixtures";
  }

  static fs::path get_checkpoint_path(const std::string& name) {
    return get_test_data_dir() / name;
  }
};

TEST_F(NordlysIntegrationTest, LoadCheckpointAndRoute) {
  auto checkpoint_path = get_checkpoint_path("valid_checkpoint_f32.json");
  ASSERT_TRUE(fs::exists(checkpoint_path)) << "Test data file not found: " << checkpoint_path;

  auto checkpoint = NordlysCheckpoint::from_json(checkpoint_path.string());
  auto result = Nordlys::from_checkpoint(std::move(checkpoint));

  ASSERT_TRUE(result) << "Failed to create Nordlys: " << result.error();

  auto engine = std::move(result.value());
  EXPECT_EQ(engine.get_n_clusters(), 3);
  EXPECT_EQ(engine.get_embedding_dim(), 4);

  auto models = engine.get_supported_models();
  EXPECT_EQ(models.size(), 3);
  EXPECT_TRUE(std::find(models.begin(), models.end(), "openai/gpt-4") != models.end());
  EXPECT_TRUE(std::find(models.begin(), models.end(), "anthropic/claude-3-opus") != models.end());
  EXPECT_TRUE(std::find(models.begin(), models.end(), "meta/llama-3-70b") != models.end());

  std::vector<float> embedding = {0.95f, 0.05f, 0.0f, 0.0f};
  EmbeddingView view{embedding.data(), embedding.size(), Device{CpuDevice{}}};
  auto response = engine.route(view);

  EXPECT_FALSE(response.selected_model.empty());
  EXPECT_EQ(response.cluster_id, 0);
  EXPECT_GE(response.cluster_distance, 0.0f);
  EXPECT_LT(response.cluster_distance, 0.2f);
}

TEST_F(NordlysIntegrationTest, RoutingSelectsByErrorRate) {
  auto checkpoint_path = get_checkpoint_path("valid_checkpoint_f32.json");
  auto checkpoint = NordlysCheckpoint::from_json(checkpoint_path.string());
  auto result = Nordlys::from_checkpoint(std::move(checkpoint));
  ASSERT_TRUE(result);

  auto engine = std::move(result.value());
  std::vector<float> embedding = {0.95f, 0.05f, 0.0f, 0.0f};

  EmbeddingView view{embedding.data(), embedding.size(), Device{CpuDevice{}}};
  auto response = engine.route(view);

  EXPECT_FALSE(response.selected_model.empty());
  EXPECT_GE(response.cluster_id, 0);
}

TEST_F(NordlysIntegrationTest, ModelFilteringWorks) {
  auto checkpoint_path = get_checkpoint_path("valid_checkpoint_f32.json");
  auto checkpoint = NordlysCheckpoint::from_json(checkpoint_path.string());
  auto result = Nordlys::from_checkpoint(std::move(checkpoint));
  ASSERT_TRUE(result);

  auto engine = std::move(result.value());
  std::vector<float> embedding = {0.95f, 0.05f, 0.0f, 0.0f};

  std::vector<std::string> filter = {"openai/gpt-4"};
  EmbeddingView view{embedding.data(), embedding.size(), Device{CpuDevice{}}};
  auto response = engine.route(view, filter);

  EXPECT_EQ(response.selected_model, "openai/gpt-4");
  EXPECT_TRUE(response.alternatives.empty());
}

TEST_F(NordlysIntegrationTest, AlternativesReturned) {
  auto checkpoint_path = get_checkpoint_path("valid_checkpoint_f32.json");
  auto checkpoint = NordlysCheckpoint::from_json(checkpoint_path.string());
  auto result = Nordlys::from_checkpoint(std::move(checkpoint));
  ASSERT_TRUE(result);

  auto engine = std::move(result.value());
  std::vector<float> embedding = {0.95f, 0.05f, 0.0f, 0.0f};
  EmbeddingView view{embedding.data(), embedding.size(), Device{CpuDevice{}}};
  auto response = engine.route(view);

  EXPECT_FALSE(response.selected_model.empty());
  EXPECT_LE(response.alternatives.size(), 2);

  for (const auto& alt : response.alternatives) {
    EXPECT_NE(alt, response.selected_model);
  }
}

TEST_F(NordlysIntegrationTest, DimensionMismatchThrows) {
  auto checkpoint_path = get_checkpoint_path("valid_checkpoint_f32.json");
  auto checkpoint = NordlysCheckpoint::from_json(checkpoint_path.string());
  auto result = Nordlys::from_checkpoint(std::move(checkpoint));
  ASSERT_TRUE(result);

  auto engine = std::move(result.value());

  std::vector<float> wrong_dim = {1.0f, 0.0f};
  EmbeddingView wrong_view{wrong_dim.data(), wrong_dim.size(), Device{CpuDevice{}}};
  EXPECT_THROW(engine.route(wrong_view), std::invalid_argument);
}

TEST_F(NordlysIntegrationTest, DifferentEmbeddingsAssignToDifferentClusters) {
  auto checkpoint_path = get_checkpoint_path("valid_checkpoint_f32.json");
  auto checkpoint = NordlysCheckpoint::from_json(checkpoint_path.string());
  auto result = Nordlys::from_checkpoint(std::move(checkpoint));
  ASSERT_TRUE(result);

  auto engine = std::move(result.value());

  std::vector<float> emb0 = {0.95f, 0.05f, 0.0f, 0.0f};
  std::vector<float> emb1 = {0.05f, 0.95f, 0.0f, 0.0f};
  std::vector<float> emb2 = {0.0f, 0.05f, 0.95f, 0.0f};

  EmbeddingView view0{emb0.data(), emb0.size(), Device{CpuDevice{}}};
  EmbeddingView view1{emb1.data(), emb1.size(), Device{CpuDevice{}}};
  EmbeddingView view2{emb2.data(), emb2.size(), Device{CpuDevice{}}};
  auto resp0 = engine.route(view0);
  auto resp1 = engine.route(view1);
  auto resp2 = engine.route(view2);

  EXPECT_EQ(resp0.cluster_id, 0);
  EXPECT_EQ(resp1.cluster_id, 1);
  EXPECT_EQ(resp2.cluster_id, 2);
}

TEST_F(NordlysIntegrationTest, NonexistentCheckpointFileThrows) {
  EXPECT_THROW(NordlysCheckpoint::from_json("/nonexistent/path/checkpoint.json"),
               std::runtime_error);
}
