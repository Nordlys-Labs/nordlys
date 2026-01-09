#include <gtest/gtest.h>

#include <nordlys_core/nordlys.hpp>
#include <vector>

// Suppress nodiscard warnings in tests since EXPECT_THROW requires ignoring return values
#pragma GCC diagnostic ignored "-Wunused-result"

// ============================================================================
// Test Fixture for Nordlys32 Tests
// ============================================================================

// Test checkpoint JSON for creating valid routers (v2.0 format)
static const char* kTestCheckpointJson = R"({
  "version": "2.0",
  "cluster_centers": [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0]
  ],
  "models": [
    {
      "model_id": "provider1/gpt-4",
      "cost_per_1m_input_tokens": 30.0,
      "cost_per_1m_output_tokens": 60.0,
      "error_rates": [0.01, 0.02, 0.015]
    },
    {
      "model_id": "provider2/llama",
      "cost_per_1m_input_tokens": 0.3,
      "cost_per_1m_output_tokens": 0.6,
      "error_rates": [0.05, 0.06, 0.055]
    }
  ],
  "embedding": {
    "model": "test-model",
    "dtype": "float32",
    "trust_remote_code": false
  },
  "clustering": {
    "n_clusters": 3,
    "random_state": 42,
    "max_iter": 300,
    "n_init": 10,
    "algorithm": "lloyd",
    "normalization": "l2"
  },
  "routing": {
    "cost_bias_min": 0.0,
    "cost_bias_max": 1.0
  },
  "metrics": {
    "silhouette_score": 0.85
  }
})";

// Test checkpoint JSON for creating double-precision routers (v2.0 format)
static const char* kTestCheckpointJsonFloat64 = R"({
  "version": "2.0",
  "cluster_centers": [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0]
  ],
  "models": [
    {
      "model_id": "provider1/gpt-4",
      "cost_per_1m_input_tokens": 30.0,
      "cost_per_1m_output_tokens": 60.0,
      "error_rates": [0.01, 0.02, 0.015]
    },
    {
      "model_id": "provider2/llama",
      "cost_per_1m_input_tokens": 0.3,
      "cost_per_1m_output_tokens": 0.6,
      "error_rates": [0.05, 0.06, 0.055]
    }
  ],
  "embedding": {
    "model": "test-model",
    "dtype": "float64",
    "trust_remote_code": false
  },
  "clustering": {
    "n_clusters": 3,
    "random_state": 42,
    "max_iter": 300,
    "n_init": 10,
    "algorithm": "lloyd",
    "normalization": "l2"
  },
  "routing": {
    "cost_bias_min": 0.0,
    "cost_bias_max": 1.0
  },
  "metrics": {
    "silhouette_score": 0.85
  }
})";

class Nordlysest : public ::testing::Test {
protected:
  // Helper to create a test router from JSON string
  Nordlys32 CreateTestRouter() {
    auto checkpoint = NordlysCheckpoint::from_json_string(kTestCheckpointJson);
    auto result = Nordlys32::from_checkpoint(std::move(checkpoint));
    if (!result) {
      throw std::runtime_error("Failed to create test router: " + result.error());
    }
    return std::move(result.value());
  }

  // Helper to create a double-precision test router
  Nordlys<double> CreateTestRouterDouble() {
    auto checkpoint = NordlysCheckpoint::from_json_string(kTestCheckpointJsonFloat64);
    auto result = Nordlys<double>::from_checkpoint(std::move(checkpoint));
    if (!result) {
      throw std::runtime_error("Failed to create test router (double): " + result.error());
    }
    return std::move(result.value());
  }
};

// ============================================================================
// Tests for Nordlys32 Basic Initialization and Properties
// ============================================================================

TEST_F(Nordlysest, BasicInitialization) {
  // Verify router can be created and basic properties are accessible
  auto router = CreateTestRouter();

  // Check that basic getters work
  EXPECT_EQ(router.get_embedding_dim(), 4);
  EXPECT_EQ(router.get_n_clusters(), 3);

  auto models = router.get_supported_models();
  EXPECT_EQ(models.size(), 2);
}

TEST_F(Nordlysest, EmbeddingDimension) {
  // Verify get_embedding_dim() returns correct value from checkpoint
  auto router = CreateTestRouter();

  // Checkpoint has 4D embeddings (3 clusters x 4 dimensions)
  EXPECT_EQ(router.get_embedding_dim(), 4);
}

TEST_F(Nordlysest, NClusters) {
  // Verify get_n_clusters() returns correct value from checkpoint
  auto router = CreateTestRouter();

  // Checkpoint has 3 clusters
  EXPECT_EQ(router.get_n_clusters(), 3);
}

TEST_F(Nordlysest, SupportedModels) {
  // Verify get_supported_models() returns all models from checkpoint
  auto router = CreateTestRouter();

  auto models = router.get_supported_models();
  EXPECT_EQ(models.size(), 2);

  // Check model IDs are correct
  EXPECT_EQ(models[0], "provider1/gpt-4");
  EXPECT_EQ(models[1], "provider2/llama");
}

// ============================================================================
// Tests for Nordlys32 Routing Functionality
// ============================================================================

TEST_F(Nordlysest, BasicRoutingWithFloat) {
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

TEST_F(Nordlysest, RoutingWithCustomCostBias) {
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

TEST_F(Nordlysest, RoutingWithFilteredModels) {
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

TEST_F(Nordlysest, RoutingWithDouble) {
  // Verify Nordlys<double> works with double precision
  auto router = CreateTestRouterDouble();

  // Create double-precision embedding
  std::vector<double> embedding = {0.95, 0.05, 0.0, 0.0};

  // Should work with double precision router
  auto response = router.route(embedding.data(), embedding.size(), 0.5f);

  EXPECT_EQ(response.cluster_id, 0);
  EXPECT_FALSE(response.selected_model.empty());
}

TEST_F(Nordlysest, FloatAndDoublePrecisionConsistency) {
  // Verify float and double routers give consistent cluster assignments
  auto router_float = CreateTestRouter();
  auto router_double = CreateTestRouterDouble();

  std::vector<float> embedding_float = {0.0f, 0.95f, 0.05f, 0.0f};
  std::vector<double> embedding_double = {0.0, 0.95, 0.05, 0.0};

  auto response_float = router_float.route(embedding_float.data(), embedding_float.size(), 0.5f);
  auto response_double
      = router_double.route(embedding_double.data(), embedding_double.size(), 0.5f);

  // Both should assign to same cluster
  EXPECT_EQ(response_float.cluster_id, response_double.cluster_id);
}

// ============================================================================
// Tests for Error Handling
// ============================================================================

TEST_F(Nordlysest, DimensionMismatchThrows) {
  // Verify route() throws std::invalid_argument on embedding dimension mismatch
  auto router = CreateTestRouter();

  // Create wrong-sized embedding (3D instead of 4D)
  std::vector<float> wrong_embedding = {0.5f, 0.5f, 0.0f};

  EXPECT_THROW(router.route(wrong_embedding.data(), wrong_embedding.size(), 0.5f),
               std::invalid_argument);
}

TEST_F(Nordlysest, DimensionMismatchErrorMessage) {
  // Verify error message contains dimension information
  auto router = CreateTestRouter();

  std::vector<float> wrong_embedding = {1.0f, 0.0f};  // 2D instead of 4D

  try {
    (void)router.route(wrong_embedding.data(), wrong_embedding.size(), 0.5f);
    FAIL() << "Expected std::invalid_argument to be thrown";
  } catch (const std::invalid_argument& e) {
    std::string msg = e.what();
    // Error message should mention dimension mismatch
    EXPECT_TRUE(msg.find("Embedding dimension mismatch") != std::string::npos
                || msg.find("dimension") != std::string::npos);
  }
}

// ============================================================================
// Tests for Model Filtering
// ============================================================================

TEST_F(Nordlysest, RoutingWithEmptyModelFilter) {
  // Empty filter should use all available models
  auto router = CreateTestRouter();

  std::vector<float> embedding = {0.95f, 0.05f, 0.0f, 0.0f};
  std::vector<std::string> empty_filter;

  auto response = router.route(embedding.data(), embedding.size(), 0.5f, empty_filter);

  EXPECT_EQ(response.cluster_id, 0);
  EXPECT_FALSE(response.selected_model.empty());
}

TEST_F(Nordlysest, RoutingWithSingleModelFilter) {
  // Filter to only one model - should always select that model
  auto router = CreateTestRouter();

  std::vector<float> embedding = {0.95f, 0.05f, 0.0f, 0.0f};
  std::vector<std::string> filter = {"provider2/llama"};

  auto response = router.route(embedding.data(), embedding.size(), 0.5f, filter);

  EXPECT_EQ(response.cluster_id, 0);
  EXPECT_EQ(response.selected_model, "provider2/llama");
}

TEST_F(Nordlysest, AlternativesReturned) {
  // Verify that alternatives are returned (all models except selected)
  auto router = CreateTestRouter();

  std::vector<float> embedding = {0.95f, 0.05f, 0.0f, 0.0f};

  auto response = router.route(embedding.data(), embedding.size(), 0.5f);

  // With 2 models, alternatives should have at most 1 (all except selected)
  EXPECT_LE(response.alternatives.size(), 1);
}

// ============================================================================
// Tests for Cluster Assignment
// ============================================================================

TEST_F(Nordlysest, DifferentEmbeddingsAssignToDifferentClusters) {
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

TEST_F(Nordlysest, ClusterDistanceIsNonNegative) {
  // Verify cluster distance is always non-negative
  auto router = CreateTestRouter();

  std::vector<float> embedding = {0.5f, 0.5f, 0.0f, 0.0f};

  auto response = router.route(embedding.data(), embedding.size(), 0.5f);

  EXPECT_GE(response.cluster_distance, 0.0f);
}

TEST_F(Nordlysest, ExactCentroidMatchHasSmallDistance) {
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

TEST_F(Nordlysest, CostBiasAffectsModelSelection) {
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

TEST_F(Nordlysest, ExtremeCostBiasValues) {
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

TEST_F(Nordlysest, ResponseContainsAllRequiredFields) {
  // Verify response has all required fields populated
  auto router = CreateTestRouter();

  std::vector<float> embedding = {0.95f, 0.05f, 0.0f, 0.0f};

  auto response = router.route(embedding.data(), embedding.size(), 0.5f);

  // Check all fields are populated
  EXPECT_FALSE(response.selected_model.empty());
  EXPECT_GE(response.cluster_id, 0);
  EXPECT_LT(response.cluster_id, 3);
  EXPECT_GE(response.cluster_distance, 0.0f);
  // alternatives can be empty, contains all models except selected
  const auto models = router.get_supported_models();
  const auto model_count = static_cast<unsigned int>(models.size());
  EXPECT_LE(response.alternatives.size(), model_count - 1u);
}

TEST_F(Nordlysest, AlternativeModelsAreDifferentFromSelected) {
  // Verify alternative models don't include the selected model
  auto router = CreateTestRouter();

  std::vector<float> embedding = {0.95f, 0.05f, 0.0f, 0.0f};

  auto response = router.route(embedding.data(), embedding.size(), 0.5f);

  for (const auto& alt : response.alternatives) {
    EXPECT_NE(alt, response.selected_model);
  }
}

// ============================================================================
// Tests for Nordlys32 Creation from Different Sources
// ============================================================================

TEST_F(Nordlysest, CreateFromJsonString) {
  // Test creating router from JSON string
  std::string json_checkpoint = R"({
    "version": "2.0",
    "cluster_centers": [[1.0, 0.0], [0.0, 1.0]],
    "models": [
      {
        "model_id": "test/model1",
        "cost_per_1m_input_tokens": 1.0,
        "cost_per_1m_output_tokens": 2.0,
        "error_rates": [0.01, 0.02]
      }
    ],
    "embedding": {"model": "test", "dtype": "float32", "trust_remote_code": false},
    "clustering": {"n_clusters": 2, "random_state": 42, "max_iter": 300, "n_init": 10, "algorithm": "lloyd", "normalization": "l2"},
    "routing": {"cost_bias_min": 0.0, "cost_bias_max": 1.0},
    "metrics": {"silhouette_score": 0.8}
  })";

  auto checkpoint = NordlysCheckpoint::from_json_string(json_checkpoint);
  auto result = Nordlys32::from_checkpoint(std::move(checkpoint));

  ASSERT_TRUE(result.has_value()) << "Failed to create router from JSON: " << result.error();

  auto& router = result.value();
  EXPECT_EQ(router.get_embedding_dim(), 2);
  EXPECT_EQ(router.get_n_clusters(), 2);
}

TEST_F(Nordlysest, CreateFromInvalidJsonStringFails) {
  // Test that invalid JSON throws an exception
  std::string invalid_json = "{ invalid json }";

  EXPECT_THROW(NordlysCheckpoint::from_json_string(invalid_json), std::exception);
}

TEST_F(Nordlysest, CreateFromNonexistentFileFails) {
  // Test that loading from nonexistent file throws an exception
  EXPECT_THROW(NordlysCheckpoint::from_json("nonexistent_file.json"), std::exception);
}

TEST_F(Nordlysest, DoublePrecisionClusterDistance) {
  // Verify RouteResult<double> preserves double precision
  auto router = CreateTestRouterDouble();

  std::vector<double> embedding = {0.95, 0.05, 0.0, 0.0};
  auto response = router.route(embedding.data(), embedding.size(), 0.5f);

  // Compile-time verification
  static_assert(std::is_same_v<decltype(response.cluster_distance), double>);

  EXPECT_EQ(response.cluster_id, 0);
  EXPECT_GE(response.cluster_distance, 0.0);
}

TEST_F(Nordlysest, DtypeMismatchValidation) {
  // Test that Nordlys<float> rejects float64 checkpoints and vice versa

  // Create float64 checkpoint
  auto float64_checkpoint = NordlysCheckpoint::from_json_string(kTestCheckpointJsonFloat64);

  // Try to create Nordlys<float> with float64 checkpoint - should fail
  auto result_float = Nordlys32::from_checkpoint(std::move(float64_checkpoint));
  EXPECT_FALSE(result_float.has_value());
  EXPECT_TRUE(result_float.error().find("requires float32 checkpoint") != std::string::npos);

  // Create float32 checkpoint
  auto float32_checkpoint = NordlysCheckpoint::from_json_string(kTestCheckpointJson);

  // Try to create Nordlys<double> with float32 checkpoint - should fail
  auto result_double = Nordlys<double>::from_checkpoint(std::move(float32_checkpoint));
  EXPECT_FALSE(result_double.has_value());
  EXPECT_TRUE(result_double.error().find("requires float64 checkpoint") != std::string::npos);
}
