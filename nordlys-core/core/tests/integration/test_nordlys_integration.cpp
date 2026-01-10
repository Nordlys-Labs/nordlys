#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <nordlys_core/checkpoint.hpp>
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

TEST_F(NordlysIntegrationTest, LoadCheckpointAndRouteFloat32) {
  auto checkpoint_path = get_checkpoint_path("valid_checkpoint_f32.json");
  ASSERT_TRUE(fs::exists(checkpoint_path)) << "Test data file not found: " << checkpoint_path;

  auto checkpoint = NordlysCheckpoint::from_json(checkpoint_path.string());
  auto result = Nordlys32::from_checkpoint(std::move(checkpoint));

  ASSERT_TRUE(result) << "Failed to create Nordlys32: " << result.error();

  auto engine = std::move(result.value());
  EXPECT_EQ(engine.get_n_clusters(), 3);
  EXPECT_EQ(engine.get_embedding_dim(), 4);

  auto models = engine.get_supported_models();
  EXPECT_EQ(models.size(), 3);
  EXPECT_TRUE(std::find(models.begin(), models.end(), "openai/gpt-4") != models.end());
  EXPECT_TRUE(std::find(models.begin(), models.end(), "anthropic/claude-3-opus") != models.end());
  EXPECT_TRUE(std::find(models.begin(), models.end(), "meta/llama-3-70b") != models.end());

  std::vector<float> embedding = {0.95f, 0.05f, 0.0f, 0.0f};
  auto response = engine.route(embedding.data(), embedding.size(), 0.5f);

  EXPECT_FALSE(response.selected_model.empty());
  EXPECT_EQ(response.cluster_id, 0);
  EXPECT_GE(response.cluster_distance, 0.0f);
  EXPECT_LT(response.cluster_distance, 0.2f);
}

TEST_F(NordlysIntegrationTest, LoadCheckpointAndRouteFloat64) {
  auto checkpoint_path = get_checkpoint_path("valid_checkpoint_f64.json");
  ASSERT_TRUE(fs::exists(checkpoint_path)) << "Test data file not found: " << checkpoint_path;

  auto checkpoint = NordlysCheckpoint::from_json(checkpoint_path.string());
  auto result = Nordlys64::from_checkpoint(std::move(checkpoint));

  ASSERT_TRUE(result) << "Failed to create Nordlys64: " << result.error();

  auto engine = std::move(result.value());
  EXPECT_EQ(engine.get_n_clusters(), 3);
  EXPECT_EQ(engine.get_embedding_dim(), 4);

  std::vector<double> embedding = {0.0, 0.95, 0.05, 0.0};
  auto response = engine.route(embedding.data(), embedding.size(), 0.5f);

  EXPECT_FALSE(response.selected_model.empty());
  EXPECT_EQ(response.cluster_id, 1);
  EXPECT_GE(response.cluster_distance, 0.0);
}

TEST_F(NordlysIntegrationTest, CostBiasAffectsModelSelection) {
  auto checkpoint_path = get_checkpoint_path("valid_checkpoint_f32.json");
  auto checkpoint = NordlysCheckpoint::from_json(checkpoint_path.string());
  auto result = Nordlys32::from_checkpoint(std::move(checkpoint));
  ASSERT_TRUE(result);

  auto engine = std::move(result.value());
  std::vector<float> embedding = {0.95f, 0.05f, 0.0f, 0.0f};

  auto response_cheap = engine.route(embedding.data(), embedding.size(), 1.0f);
  auto response_quality = engine.route(embedding.data(), embedding.size(), 0.0f);

  EXPECT_FALSE(response_cheap.selected_model.empty());
  EXPECT_FALSE(response_quality.selected_model.empty());

  EXPECT_EQ(response_cheap.cluster_id, response_quality.cluster_id);

  EXPECT_EQ(response_cheap.selected_model, "meta/llama-3-70b");

  EXPECT_TRUE(response_quality.selected_model == "openai/gpt-4"
              || response_quality.selected_model == "anthropic/claude-3-opus");
}

TEST_F(NordlysIntegrationTest, ModelFilteringWorks) {
  auto checkpoint_path = get_checkpoint_path("valid_checkpoint_f32.json");
  auto checkpoint = NordlysCheckpoint::from_json(checkpoint_path.string());
  auto result = Nordlys32::from_checkpoint(std::move(checkpoint));
  ASSERT_TRUE(result);

  auto engine = std::move(result.value());
  std::vector<float> embedding = {0.95f, 0.05f, 0.0f, 0.0f};

  std::vector<std::string> filter = {"openai/gpt-4"};
  auto response = engine.route(embedding.data(), embedding.size(), 0.5f, filter);

  EXPECT_EQ(response.selected_model, "openai/gpt-4");
  EXPECT_TRUE(response.alternatives.empty());
}

TEST_F(NordlysIntegrationTest, AlternativesReturned) {
  auto checkpoint_path = get_checkpoint_path("valid_checkpoint_f32.json");
  auto checkpoint = NordlysCheckpoint::from_json(checkpoint_path.string());
  auto result = Nordlys32::from_checkpoint(std::move(checkpoint));
  ASSERT_TRUE(result);

  auto engine = std::move(result.value());
  std::vector<float> embedding = {0.95f, 0.05f, 0.0f, 0.0f};
  auto response = engine.route(embedding.data(), embedding.size(), 0.5f);

  EXPECT_FALSE(response.selected_model.empty());
  EXPECT_LE(response.alternatives.size(), 2);

  for (const auto& alt : response.alternatives) {
    EXPECT_NE(alt, response.selected_model);
  }
}

TEST_F(NordlysIntegrationTest, DtypeMismatchFails) {
  auto checkpoint_path = get_checkpoint_path("valid_checkpoint_f64.json");
  auto checkpoint = NordlysCheckpoint::from_json(checkpoint_path.string());

  auto result = Nordlys32::from_checkpoint(std::move(checkpoint));

  EXPECT_FALSE(result);
  EXPECT_TRUE(result.error().find("float32") != std::string::npos);
}

TEST_F(NordlysIntegrationTest, DimensionMismatchThrows) {
  auto checkpoint_path = get_checkpoint_path("valid_checkpoint_f32.json");
  auto checkpoint = NordlysCheckpoint::from_json(checkpoint_path.string());
  auto result = Nordlys32::from_checkpoint(std::move(checkpoint));
  ASSERT_TRUE(result);

  auto engine = std::move(result.value());

  std::vector<float> wrong_dim = {1.0f, 0.0f};
  EXPECT_THROW(engine.route(wrong_dim.data(), wrong_dim.size(), 0.5f), std::invalid_argument);
}

TEST_F(NordlysIntegrationTest, DifferentEmbeddingsAssignToDifferentClusters) {
  auto checkpoint_path = get_checkpoint_path("valid_checkpoint_f32.json");
  auto checkpoint = NordlysCheckpoint::from_json(checkpoint_path.string());
  auto result = Nordlys32::from_checkpoint(std::move(checkpoint));
  ASSERT_TRUE(result);

  auto engine = std::move(result.value());

  std::vector<float> emb0 = {0.95f, 0.05f, 0.0f, 0.0f};
  std::vector<float> emb1 = {0.05f, 0.95f, 0.0f, 0.0f};
  std::vector<float> emb2 = {0.0f, 0.05f, 0.95f, 0.0f};

  auto resp0 = engine.route(emb0.data(), emb0.size(), 0.5f);
  auto resp1 = engine.route(emb1.data(), emb1.size(), 0.5f);
  auto resp2 = engine.route(emb2.data(), emb2.size(), 0.5f);

  EXPECT_EQ(resp0.cluster_id, 0);
  EXPECT_EQ(resp1.cluster_id, 1);
  EXPECT_EQ(resp2.cluster_id, 2);
}

TEST_F(NordlysIntegrationTest, NonexistentCheckpointFileThrows) {
  EXPECT_THROW(NordlysCheckpoint::from_json("/nonexistent/path/checkpoint.json"),
               std::runtime_error);
}
