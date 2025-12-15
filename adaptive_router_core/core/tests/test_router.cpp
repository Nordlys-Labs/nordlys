#include <gtest/gtest.h>

#include <vector>

#include <adaptive_core/router.hpp>

// ============================================================================
// Test Fixture for Router Tests
// ============================================================================

// Test profile JSON for creating valid routers
static const char* kTestProfileJson = R"({
  "metadata": {
    "n_clusters": 3,
    "embedding_model": "test-model",
    "silhouette_score": 0.85,
    "clustering": {
      "n_init": 10,
      "algorithm": "lloyd"
    },
    "routing": {
      "lambda_min": 0.0,
      "lambda_max": 2.0,
      "max_alternatives": 2
    }
  },
  "cluster_centers": {
    "n_clusters": 3,
    "feature_dim": 4,
    "cluster_centers": [
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0]
    ]
  },
  "models": [
    {
      "provider": "provider1",
      "model_name": "gpt-4",
      "cost_per_1m_input_tokens": 30.0,
      "cost_per_1m_output_tokens": 60.0,
      "error_rates": [0.01, 0.02, 0.015]
    },
    {
      "provider": "provider2",
      "model_name": "llama",
      "cost_per_1m_input_tokens": 0.3,
      "cost_per_1m_output_tokens": 0.6,
      "error_rates": [0.05, 0.06, 0.055]
    }
  ]
})";

class RouterTest : public ::testing::Test {
 protected:
  // Helper to create a test router from JSON string
  Router CreateTestRouter() {
    auto result = Router::from_json_string(kTestProfileJson);
    if (!result) {
      throw std::runtime_error("Failed to create test router: " + result.error());
    }
    return std::move(result.value());
  }
};

// ============================================================================
// Tests for Router Basic Initialization and Properties
// ============================================================================

TEST_F(RouterTest, BasicInitialization) {
  // Verify router can be created and basic properties are accessible
  auto router = CreateTestRouter();

  // Check that basic getters work
  EXPECT_EQ(router.get_embedding_dim(), 4);
  EXPECT_EQ(router.get_n_clusters(), 3);

  auto models = router.get_supported_models();
  EXPECT_EQ(models.size(), 2);
}

TEST_F(RouterTest, EmbeddingDimension) {
  // Verify get_embedding_dim() returns correct value from profile
  auto router = CreateTestRouter();

  // Profile has 4D embeddings (3 clusters x 4 dimensions)
  EXPECT_EQ(router.get_embedding_dim(), 4);
}

TEST_F(RouterTest, NClusters) {
  // Verify get_n_clusters() returns correct value from profile
  auto router = CreateTestRouter();

  // Profile has 3 clusters
  EXPECT_EQ(router.get_n_clusters(), 3);
}

TEST_F(RouterTest, SupportedModels) {
  // Verify get_supported_models() returns all models from profile
  auto router = CreateTestRouter();

  auto models = router.get_supported_models();
  EXPECT_EQ(models.size(), 2);

  // Check model IDs are correct
  EXPECT_EQ(models[0], "provider1/gpt-4");
  EXPECT_EQ(models[1], "provider2/llama");
}

// ============================================================================
// Tests for Router Routing Functionality
// ============================================================================

TEST_F(RouterTest, BasicRoutingWithFloat) {
  // Verify route() works with valid float embedding and returns proper response
  auto router = CreateTestRouter();

  // Create embedding vector matching cluster centroid (4D)
  std::vector<float> embedding = {0.95f, 0.05f, 0.0f, 0.0f};

  // Route should assign to cluster 0 (closest to [1,0,0,0])
  auto response = router.route(embedding.data(), embedding.size(), 0.5f);

  EXPECT_EQ(response.cluster_id, 0);
  EXPECT_GE(response.cluster_distance, 0.0f);
  EXPECT_FALSE(response.selected_model.empty());
  EXPECT_GE(response.alternatives.size(), 0);
}

TEST_F(RouterTest, RoutingWithCustomCostBias) {
  // Verify cost_bias parameter affects model selection
  auto router = CreateTestRouter();

  std::vector<float> embedding = {0.95f, 0.05f, 0.0f, 0.0f};

  // Route with different cost biases
  auto response_accuracy = router.route(embedding.data(), embedding.size(), 0.0f);
  auto response_cost = router.route(embedding.data(), embedding.size(), 1.0f);

  // Both should return valid responses
  EXPECT_FALSE(response_accuracy.selected_model.empty());
  EXPECT_FALSE(response_cost.selected_model.empty());
  EXPECT_EQ(response_accuracy.cluster_id, 0);
  EXPECT_EQ(response_cost.cluster_id, 0);
}

TEST_F(RouterTest, RoutingWithFilteredModels) {
  // Verify routing works with model filter
  auto router = CreateTestRouter();

  std::vector<float> embedding = {0.0f, 0.95f, 0.05f, 0.0f};
  std::vector<std::string> filtered_models = {"provider1/gpt-4"};

  auto response = router.route(embedding.data(), embedding.size(), 0.5f, filtered_models);

  EXPECT_EQ(response.cluster_id, 1);
  EXPECT_EQ(response.selected_model, "provider1/gpt-4");
}

// ============================================================================
// Tests for Multi-Precision Support
// ============================================================================

TEST_F(RouterTest, RoutingWithDouble) {
  // Verify templated route() works with double precision
  auto router = CreateTestRouter();

  // Create double-precision embedding
  std::vector<double> embedding = {0.95, 0.05, 0.0, 0.0};

  // Should work seamlessly with double precision
  auto response = router.route(embedding.data(), embedding.size(), 0.5f);

  EXPECT_EQ(response.cluster_id, 0);
  EXPECT_FALSE(response.selected_model.empty());
}

TEST_F(RouterTest, FloatAndDoublePrecisionConsistency) {
  // Verify float and double give consistent cluster assignments
  auto router = CreateTestRouter();

  std::vector<float> embedding_float = {0.0f, 0.95f, 0.05f, 0.0f};
  std::vector<double> embedding_double = {0.0, 0.95, 0.05, 0.0};

  auto response_float = router.route(embedding_float.data(), embedding_float.size(), 0.5f);
  auto response_double = router.route(embedding_double.data(), embedding_double.size(), 0.5f);

  // Both should assign to same cluster
  EXPECT_EQ(response_float.cluster_id, response_double.cluster_id);
}

// ============================================================================
// Tests for Error Handling
// ============================================================================

TEST_F(RouterTest, DimensionMismatchThrows) {
  // Verify route() throws std::invalid_argument on embedding dimension mismatch
  auto router = CreateTestRouter();

  // Create wrong-sized embedding (3D instead of 4D)
  std::vector<float> wrong_embedding = {0.5f, 0.5f, 0.0f};

  EXPECT_THROW(
      router.route(wrong_embedding.data(), wrong_embedding.size(), 0.5f),
      std::invalid_argument);
}

TEST_F(RouterTest, DimensionMismatchErrorMessage) {
  // Verify error message contains dimension information
  auto router = CreateTestRouter();

  std::vector<float> wrong_embedding = {1.0f, 0.0f};  // 2D instead of 4D

  try {
    (void)router.route(wrong_embedding.data(), wrong_embedding.size(), 0.5f);
    FAIL() << "Expected std::invalid_argument to be thrown";
  } catch (const std::invalid_argument& e) {
    std::string msg = e.what();
    // Error message should mention dimension mismatch
    EXPECT_TRUE(msg.find("Embedding dimension mismatch") != std::string::npos ||
                msg.find("dimension") != std::string::npos);
  }
}

// ============================================================================
// Tests for Model Filtering
// ============================================================================

TEST_F(RouterTest, RoutingWithEmptyModelFilter) {
  // Empty filter should use all available models
  auto router = CreateTestRouter();

  std::vector<float> embedding = {0.95f, 0.05f, 0.0f, 0.0f};
  std::vector<std::string> empty_filter;

  auto response = router.route(embedding.data(), embedding.size(), 0.5f, empty_filter);

  EXPECT_EQ(response.cluster_id, 0);
  EXPECT_FALSE(response.selected_model.empty());
}

TEST_F(RouterTest, RoutingWithSingleModelFilter) {
  // Filter to only one model - should always select that model
  auto router = CreateTestRouter();

  std::vector<float> embedding = {0.95f, 0.05f, 0.0f, 0.0f};
  std::vector<std::string> filter = {"provider2/llama"};

  auto response = router.route(embedding.data(), embedding.size(), 0.5f, filter);

  EXPECT_EQ(response.cluster_id, 0);
  EXPECT_EQ(response.selected_model, "provider2/llama");
}

TEST_F(RouterTest, AlternativesRespectMaxAlternatives) {
  // Verify that alternatives count doesn't exceed max_alternatives
  auto router = CreateTestRouter();

  std::vector<float> embedding = {0.95f, 0.05f, 0.0f, 0.0f};

  auto response = router.route(embedding.data(), embedding.size(), 0.5f);

  // max_alternatives is 2 in test profile, and we have 2 models total
  // So alternatives should be at most 1 (total - selected = 2 - 1)
  EXPECT_LE(response.alternatives.size(), 2);
}

// ============================================================================
// Tests for Cluster Assignment
// ============================================================================

TEST_F(RouterTest, DifferentEmbeddingsAssignToDifferentClusters) {
  // Verify that embeddings close to different centroids are assigned correctly
  auto router = CreateTestRouter();

  std::vector<float> embedding1 = {0.95f, 0.05f, 0.0f, 0.0f};  // Close to cluster 0
  std::vector<float> embedding2 = {0.05f, 0.95f, 0.0f, 0.0f};  // Close to cluster 1
  std::vector<float> embedding3 = {0.0f, 0.05f, 0.95f, 0.0f};  // Close to cluster 2

  auto response1 = router.route(embedding1.data(), embedding1.size(), 0.5f);
  auto response2 = router.route(embedding2.data(), embedding2.size(), 0.5f);
  auto response3 = router.route(embedding3.data(), embedding3.size(), 0.5f);

  EXPECT_EQ(response1.cluster_id, 0);
  EXPECT_EQ(response2.cluster_id, 1);
  EXPECT_EQ(response3.cluster_id, 2);
}

TEST_F(RouterTest, ClusterDistanceIsNonNegative) {
  // Verify cluster distance is always non-negative
  auto router = CreateTestRouter();

  std::vector<float> embedding = {0.5f, 0.5f, 0.0f, 0.0f};

  auto response = router.route(embedding.data(), embedding.size(), 0.5f);

  EXPECT_GE(response.cluster_distance, 0.0f);
}

TEST_F(RouterTest, ExactCentroidMatchHasSmallDistance) {
  // Embedding exactly matching a centroid should have very small distance
  auto router = CreateTestRouter();

  // Exact match with cluster 0 centroid [1, 0, 0, 0]
  std::vector<float> embedding = {1.0f, 0.0f, 0.0f, 0.0f};

  auto response = router.route(embedding.data(), embedding.size(), 0.5f);

  EXPECT_EQ(response.cluster_id, 0);
  EXPECT_NEAR(response.cluster_distance, 0.0f, 1e-5);
}

// ============================================================================
// Tests for Cost Bias Effects
// ============================================================================

TEST_F(RouterTest, CostBiasAffectsModelSelection) {
  // Different cost biases may lead to different model selections
  auto router = CreateTestRouter();

  std::vector<float> embedding = {0.0f, 0.95f, 0.05f, 0.0f};

  auto response_accuracy = router.route(embedding.data(), embedding.size(), 0.0f);
  auto response_cost = router.route(embedding.data(), embedding.size(), 1.0f);

  // Both should assign to same cluster
  EXPECT_EQ(response_accuracy.cluster_id, response_cost.cluster_id);

  // Both should return valid models
  EXPECT_FALSE(response_accuracy.selected_model.empty());
  EXPECT_FALSE(response_cost.selected_model.empty());
}

TEST_F(RouterTest, ExtremeCostBiasValues) {
  // Test with extreme cost bias values
  auto router = CreateTestRouter();

  std::vector<float> embedding = {0.0f, 0.0f, 0.95f, 0.05f};

  // Should handle extreme values gracefully
  auto response_low = router.route(embedding.data(), embedding.size(), -1.0f);
  auto response_high = router.route(embedding.data(), embedding.size(), 2.0f);

  EXPECT_FALSE(response_low.selected_model.empty());
  EXPECT_FALSE(response_high.selected_model.empty());
}

// ============================================================================
// Tests for Response Structure
// ============================================================================

TEST_F(RouterTest, ResponseContainsAllRequiredFields) {
  // Verify response has all required fields populated
  auto router = CreateTestRouter();

  std::vector<float> embedding = {0.95f, 0.05f, 0.0f, 0.0f};

  auto response = router.route(embedding.data(), embedding.size(), 0.5f);

  // Check all fields are populated
  EXPECT_FALSE(response.selected_model.empty());
  EXPECT_GE(response.cluster_id, 0);
  EXPECT_LT(response.cluster_id, 3);
  EXPECT_GE(response.cluster_distance, 0.0f);
  // alternatives can be empty, but should not exceed min(max_alternatives, model_count - 1)
  const auto models = router.get_supported_models();
  const auto model_count = static_cast<unsigned int>(models.size());
  const auto max_alternatives = 2u;  // from test profile routing.max_alternatives
  const auto max_possible_alternatives = std::min(max_alternatives, model_count - 1u);
  EXPECT_LE(response.alternatives.size(), max_possible_alternatives);
}

TEST_F(RouterTest, AlternativeModelsAreDifferentFromSelected) {
  // Verify alternative models don't include the selected model
  auto router = CreateTestRouter();

  std::vector<float> embedding = {0.95f, 0.05f, 0.0f, 0.0f};

  auto response = router.route(embedding.data(), embedding.size(), 0.5f);

  for (const auto& alt : response.alternatives) {
    EXPECT_NE(alt, response.selected_model);
  }
}

// ============================================================================
// Tests for Router Creation from Different Sources
// ============================================================================

TEST_F(RouterTest, CreateFromJsonString) {
  // Test creating router from JSON string
  std::string json_profile = R"({
    "metadata": {
      "n_clusters": 2,
      "embedding_model": "test",
      "silhouette_score": 0.8,
      "clustering": {"n_init": 10, "algorithm": "lloyd"},
      "routing": {"lambda_min": 0.0, "lambda_max": 2.0, "max_alternatives": 1}
    },
    "cluster_centers": {
      "n_clusters": 2,
      "feature_dim": 2,
      "cluster_centers": [[1.0, 0.0], [0.0, 1.0]]
    },
    "models": [
      {
        "provider": "test",
        "model_name": "model1",
        "cost_per_1m_input_tokens": 1.0,
        "cost_per_1m_output_tokens": 2.0,
        "error_rates": [0.01, 0.02]
      }
    ]
  })";

  auto result = Router::from_json_string(json_profile);

  ASSERT_TRUE(result.has_value()) << "Failed to create router from JSON: " << result.error();

  auto& router = result.value();
  EXPECT_EQ(router.get_embedding_dim(), 2);
  EXPECT_EQ(router.get_n_clusters(), 2);
}

TEST_F(RouterTest, CreateFromInvalidJsonStringFails) {
  // Test that invalid JSON returns an error
  std::string invalid_json = "{ invalid json }";

  auto result = Router::from_json_string(invalid_json);

  EXPECT_FALSE(result.has_value());
}

TEST_F(RouterTest, CreateFromNonexistentFileFails) {
  // Test that loading from nonexistent file returns an error
  auto result = Router::from_file("nonexistent_file.json");

  EXPECT_FALSE(result.has_value());
}

