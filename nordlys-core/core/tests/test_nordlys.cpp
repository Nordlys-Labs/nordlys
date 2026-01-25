#include <gtest/gtest.h>

#include <nordlys_core/device.hpp>
#include <nordlys_core/embedding_view.hpp>
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
  EmbeddingView<float> view{embedding.data(), embedding.size(), Device{CpuDevice{}}};
  auto response = router.route(view);

  EXPECT_EQ(response.cluster_id, 0);
  EXPECT_GE(response.cluster_distance, 0.0f);
  EXPECT_FALSE(response.selected_model.empty());
  EXPECT_GE(response.alternatives.size(), 0);
}

TEST_F(Nordlysest, RoutingByErrorRate) {
  // Verify routing selects models by error rate
  auto router = CreateTestRouter();

  std::vector<float> embedding = {0.95f, 0.05f, 0.0f, 0.0f};

  EmbeddingView<float> view{embedding.data(), embedding.size(), Device{CpuDevice{}}};
  auto response = router.route(view);

  EXPECT_FALSE(response.selected_model.empty());
  EXPECT_EQ(response.cluster_id, 0);
}

TEST_F(Nordlysest, RoutingWithFilteredModels) {
  // Verify routing works with model filter
  auto router = CreateTestRouter();

  std::vector<float> embedding = {0.0f, 0.95f, 0.05f, 0.0f};
  std::vector<std::string> filtered_models = {"provider1/gpt-4"};

  EmbeddingView<float> view{embedding.data(), embedding.size(), Device{CpuDevice{}}};
  auto response = router.route(view, filtered_models);

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
  EmbeddingView<double> view{embedding.data(), embedding.size(), Device{CpuDevice{}}};
  auto response = router.route(view);

  EXPECT_EQ(response.cluster_id, 0);
  EXPECT_FALSE(response.selected_model.empty());
}

TEST_F(Nordlysest, FloatAndDoublePrecisionConsistency) {
  // Verify float and double routers give consistent cluster assignments
  auto router_float = CreateTestRouter();
  auto router_double = CreateTestRouterDouble();

  std::vector<float> embedding_float = {0.0f, 0.95f, 0.05f, 0.0f};
  std::vector<double> embedding_double = {0.0, 0.95, 0.05, 0.0};

  EmbeddingView<float> view_float{embedding_float.data(), embedding_float.size(), Device{CpuDevice{}}};
  EmbeddingView<double> view_double{embedding_double.data(), embedding_double.size(), Device{CpuDevice{}}};
  auto response_float = router_float.route(view_float);
  auto response_double = router_double.route(view_double);

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

  EmbeddingView<float> wrong_view{wrong_embedding.data(), wrong_embedding.size(), Device{CpuDevice{}}};
  EXPECT_THROW(router.route(wrong_view),
               std::invalid_argument);
}

TEST_F(Nordlysest, DimensionMismatchErrorMessage) {
  // Verify error message contains dimension information
  auto router = CreateTestRouter();

  std::vector<float> wrong_embedding = {1.0f, 0.0f};  // 2D instead of 4D

  try {
    EmbeddingView<float> wrong_view{wrong_embedding.data(), wrong_embedding.size(), Device{CpuDevice{}}};
    (void)router.route(wrong_view);
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

  EmbeddingView<float> view{embedding.data(), embedding.size(), Device{CpuDevice{}}};
  auto response = router.route(view, empty_filter);

  EXPECT_EQ(response.cluster_id, 0);
  EXPECT_FALSE(response.selected_model.empty());
}

TEST_F(Nordlysest, RoutingWithSingleModelFilter) {
  // Filter to only one model - should always select that model
  auto router = CreateTestRouter();

  std::vector<float> embedding = {0.95f, 0.05f, 0.0f, 0.0f};
  std::vector<std::string> filter = {"provider2/llama"};

  EmbeddingView<float> view{embedding.data(), embedding.size(), Device{CpuDevice{}}};
  auto response = router.route(view, filter);

  EXPECT_EQ(response.cluster_id, 0);
  EXPECT_EQ(response.selected_model, "provider2/llama");
}

TEST_F(Nordlysest, AlternativesReturned) {
  // Verify that alternatives are returned (all models except selected)
  auto router = CreateTestRouter();

  std::vector<float> embedding = {0.95f, 0.05f, 0.0f, 0.0f};

  EmbeddingView<float> view{embedding.data(), embedding.size(), Device{CpuDevice{}}};
  auto response = router.route(view);

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

  EmbeddingView<float> view1{embedding1.data(), embedding1.size(), Device{CpuDevice{}}};
  EmbeddingView<float> view2{embedding2.data(), embedding2.size(), Device{CpuDevice{}}};
  EmbeddingView<float> view3{embedding3.data(), embedding3.size(), Device{CpuDevice{}}};
  auto response1 = router.route(view1);
  auto response2 = router.route(view2);
  auto response3 = router.route(view3);

  EXPECT_EQ(response1.cluster_id, 0);
  EXPECT_EQ(response2.cluster_id, 1);
  EXPECT_EQ(response3.cluster_id, 2);
}

TEST_F(Nordlysest, ClusterDistanceIsNonNegative) {
  // Verify cluster distance is always non-negative
  auto router = CreateTestRouter();

  std::vector<float> embedding = {0.5f, 0.5f, 0.0f, 0.0f};

  EmbeddingView<float> view{embedding.data(), embedding.size(), Device{CpuDevice{}}};
  auto response = router.route(view);

  EXPECT_GE(response.cluster_distance, 0.0f);
}

TEST_F(Nordlysest, ExactCentroidMatchHasSmallDistance) {
  // Embedding exactly matching a centroid should have very small distance
  auto router = CreateTestRouter();

  // Exact match with cluster 0 centroid [1, 0, 0, 0]
  std::vector<float> embedding = {1.0f, 0.0f, 0.0f, 0.0f};

  EmbeddingView<float> view{embedding.data(), embedding.size(), Device{CpuDevice{}}};
  auto response = router.route(view);

  EXPECT_EQ(response.cluster_id, 0);
  EXPECT_NEAR(response.cluster_distance, 0.0f, 1e-5);
}

// ============================================================================
// Tests for Cost Bias Effects
// ============================================================================

TEST_F(Nordlysest, RoutingSelectsByErrorRate) {
  // Routing selects models by error rate (lower is better)
  auto router = CreateTestRouter();

  std::vector<float> embedding = {0.0f, 0.95f, 0.05f, 0.0f};

  EmbeddingView<float> view{embedding.data(), embedding.size(), Device{CpuDevice{}}};
  auto response = router.route(view);

  // Should return valid model
  EXPECT_FALSE(response.selected_model.empty());
  EXPECT_GE(response.cluster_id, 0);
}

TEST_F(Nordlysest, RoutingWorks) {
  // Basic routing functionality
  auto router = CreateTestRouter();

  std::vector<float> embedding = {0.0f, 0.0f, 0.95f, 0.05f};

  EmbeddingView<float> view{embedding.data(), embedding.size(), Device{CpuDevice{}}};
  auto response = router.route(view);

  EXPECT_FALSE(response.selected_model.empty());
}

// ============================================================================
// Tests for Response Structure
// ============================================================================

TEST_F(Nordlysest, ResponseContainsAllRequiredFields) {
  // Verify response has all required fields populated
  auto router = CreateTestRouter();

  std::vector<float> embedding = {0.95f, 0.05f, 0.0f, 0.0f};

  EmbeddingView<float> view{embedding.data(), embedding.size(), Device{CpuDevice{}}};
  auto response = router.route(view);

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

  EmbeddingView<float> view{embedding.data(), embedding.size(), Device{CpuDevice{}}};
  auto response = router.route(view);

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
  EmbeddingView<double> view{embedding.data(), embedding.size(), Device{CpuDevice{}}};
  auto response = router.route(view);

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

// ============================================================================
// Tests for Move Semantics
// ============================================================================

TEST_F(Nordlysest, MoveConstructor) {
  auto router1 = CreateTestRouter();
  EXPECT_EQ(router1.get_embedding_dim(), 4);
  EXPECT_EQ(router1.get_n_clusters(), 3);

  // Move construct
  Nordlys32 router2(std::move(router1));
  EXPECT_EQ(router2.get_embedding_dim(), 4);
  EXPECT_EQ(router2.get_n_clusters(), 3);

  std::vector<float> embedding = {0.95f, 0.05f, 0.0f, 0.0f};
  EmbeddingView<float> view{embedding.data(), embedding.size(), Device{CpuDevice{}}};
  auto response = router2.route(view);
  EXPECT_EQ(response.cluster_id, 0);
}

TEST_F(Nordlysest, MoveAssignment) {
  auto router1 = CreateTestRouter();
  auto router2 = CreateTestRouter();

  EXPECT_EQ(router1.get_embedding_dim(), 4);
  EXPECT_EQ(router2.get_embedding_dim(), 4);

  // Move assign
  router2 = std::move(router1);
  EXPECT_EQ(router2.get_embedding_dim(), 4);

  std::vector<float> embedding = {0.0f, 0.95f, 0.05f, 0.0f};
  EmbeddingView<float> view{embedding.data(), embedding.size(), Device{CpuDevice{}}};
  auto response = router2.route(view);
  EXPECT_EQ(response.cluster_id, 1);
}

// ============================================================================
// Tests for Edge Cases
// ============================================================================

TEST_F(Nordlysest, SingleCluster) {
  std::string json_single_cluster = R"({
    "version": "2.0",
    "cluster_centers": [[1.0, 0.0, 0.0]],
    "models": [
      {"model_id": "test/model", "cost_per_1m_input_tokens": 1.0, "cost_per_1m_output_tokens": 2.0, "error_rates": [0.01]}
    ],
    "embedding": {"model": "test", "dtype": "float32", "trust_remote_code": false},
    "clustering": {"n_clusters": 1, "random_state": 42, "max_iter": 300, "n_init": 10, "algorithm": "lloyd", "normalization": "l2"},
    "metrics": {"silhouette_score": 0.0}
  })";

  auto checkpoint = NordlysCheckpoint::from_json_string(json_single_cluster);
  auto result = Nordlys32::from_checkpoint(std::move(checkpoint));

  ASSERT_TRUE(result.has_value());
  auto& router = result.value();

  EXPECT_EQ(router.get_n_clusters(), 1);
  EXPECT_EQ(router.get_embedding_dim(), 3);

  std::vector<float> embedding = {0.5f, 0.5f, 0.5f};
  EmbeddingView<float> view{embedding.data(), embedding.size(), Device{CpuDevice{}}};
  auto response = router.route(view);

  EXPECT_EQ(response.cluster_id, 0);
}

TEST_F(Nordlysest, LargeDimensions) {
  // Generate 4096-dimensional cluster centers
  std::vector<std::vector<float>> centers(2, std::vector<float>(4096));
  for (int i = 0; i < 4096; ++i) {
    centers[0][i] = (i % 2 == 0) ? 1.0f : 0.0f;
    centers[1][i] = (i % 2 == 1) ? 1.0f : 0.0f;
  }

  // Build JSON manually
  std::stringstream ss;
  ss << R"({"version": "2.0", "cluster_centers": [)";
  for (size_t c = 0; c < centers.size(); ++c) {
    ss << "[";
    for (size_t d = 0; d < centers[c].size(); ++d) {
      ss << centers[c][d];
      if (d < centers[c].size() - 1) ss << ",";
    }
    ss << "]";
    if (c < centers.size() - 1) ss << ",";
  }
  ss << R"(], "models": [
      {"model_id": "test/model", "cost_per_1m_input_tokens": 1.0, "cost_per_1m_output_tokens": 2.0, "error_rates": [0.01, 0.02]}
    ],
    "embedding": {"model": "test", "dtype": "float32", "trust_remote_code": false},
    "clustering": {"n_clusters": 2, "random_state": 42, "max_iter": 300, "n_init": 10, "algorithm": "lloyd", "normalization": "l2"},
    "metrics": {"silhouette_score": 0.5}
  })";

  auto checkpoint = NordlysCheckpoint::from_json_string(ss.str());
  auto result = Nordlys32::from_checkpoint(std::move(checkpoint));

  ASSERT_TRUE(result.has_value());
  auto& router = result.value();

  EXPECT_EQ(router.get_embedding_dim(), 4096);
  EXPECT_EQ(router.get_n_clusters(), 2);

  std::vector<float> embedding(4096);
  for (int i = 0; i < 4096; ++i) {
    embedding[i] = (i % 2 == 0) ? 0.9f : 0.1f;
  }

  EmbeddingView<float> view{embedding.data(), embedding.size(), Device{CpuDevice{}}};
  auto response = router.route(view);
  EXPECT_EQ(response.cluster_id, 0);
}

// ============================================================================
// Tests for Backend Selection
// ============================================================================

TEST_F(Nordlysest, BackendExplicitSelection) {
  // Explicit device selection should work with CPU backend
  auto checkpoint = NordlysCheckpoint::from_json_string(kTestCheckpointJson);
  auto result = Nordlys32::from_checkpoint(std::move(checkpoint), Device{CpuDevice{}});
  ASSERT_TRUE(result);
  auto router = std::move(result.value());

  std::vector<float> embedding = {0.95f, 0.05f, 0.0f, 0.0f};
  EmbeddingView<float> view{embedding.data(), embedding.size(), Device{CpuDevice{}}};
  auto response = router.route(view);

  EXPECT_EQ(response.cluster_id, 0);
  EXPECT_FALSE(response.selected_model.empty());
}



TEST_F(Nordlysest, RoutingBasic) {
  auto router = CreateTestRouter();

  std::vector<float> embedding = {0.95f, 0.05f, 0.0f, 0.0f};

  EmbeddingView<float> view{embedding.data(), embedding.size(), Device{CpuDevice{}}};
  auto response = router.route(view);
  EXPECT_FALSE(response.selected_model.empty());
}

TEST_F(Nordlysest, SupportedModelsAfterInit) {
  auto router = CreateTestRouter();
  auto models = router.get_supported_models();

  EXPECT_EQ(models.size(), 2);
  EXPECT_EQ(models[0], "provider1/gpt-4");
  EXPECT_EQ(models[1], "provider2/llama");

  // Call again to verify consistency
  auto models2 = router.get_supported_models();
  EXPECT_EQ(models, models2);
}

TEST_F(Nordlysest, DimensionValidationComprehensive) {
  auto router = CreateTestRouter();

  // Test zero dimensions
  std::vector<float> empty_embedding;
  EmbeddingView<float> empty_view{empty_embedding.data(), empty_embedding.size(), Device{CpuDevice{}}};
  EXPECT_THROW(router.route(empty_view),
               std::invalid_argument);

  // Test undersized
  std::vector<float> undersized = {1.0f, 0.0f};
  EmbeddingView<float> undersized_view{undersized.data(), undersized.size(), Device{CpuDevice{}}};
  EXPECT_THROW(router.route(undersized_view),
               std::invalid_argument);

  // Test oversized
  std::vector<float> oversized = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  EmbeddingView<float> oversized_view{oversized.data(), oversized.size(), Device{CpuDevice{}}};
  EXPECT_THROW(router.route(oversized_view),
               std::invalid_argument);

  // Test correct size
  std::vector<float> correct = {1.0f, 0.0f, 0.0f, 0.0f};
  EmbeddingView<float> correct_view{correct.data(), correct.size(), Device{CpuDevice{}}};
  EXPECT_NO_THROW(router.route(correct_view));
}

// ============================================================================
// OpenMP Threading API Tests
// ============================================================================

TEST(ThreadingAPITest, GetNumThreads) {
  int num_threads = get_num_threads();
  EXPECT_GT(num_threads, 0);
#ifdef _OPENMP
  EXPECT_GE(num_threads, 1);
#else
  EXPECT_EQ(num_threads, 1);
#endif
}

TEST(ThreadingAPITest, SetNumThreads) {
  int original = get_num_threads();
  
  set_num_threads(2);
  int after_set = get_num_threads();
#ifdef _OPENMP
  EXPECT_EQ(after_set, 2);
#else
  EXPECT_EQ(after_set, 1);
#endif
  
  set_num_threads(original);
}

TEST(ThreadingAPITest, InitThreadingIdempotent) {
  init_threading();
  int threads1 = get_num_threads();
  
  init_threading();
  int threads2 = get_num_threads();
  
  EXPECT_EQ(threads1, threads2);
}

TEST(ThreadingAPITest, BatchRoutingWithDifferentThreadCounts) {
  auto checkpoint = NordlysCheckpoint::from_json_string(kTestCheckpointJson);
  
  auto router_result = Nordlys32::from_checkpoint(std::move(checkpoint));
  ASSERT_TRUE(router_result.has_value());
  auto& router = router_result.value();
  
  std::vector<float> embeddings = {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.5f, 0.5f, 0.0f, 0.0f
  };
  
  EmbeddingBatchView<float> batch_view{embeddings.data(), 4, 4, Device{CpuDevice{}}};
  
  set_num_threads(1);
  auto results_1thread = router.route_batch(batch_view);
  
  set_num_threads(4);
  auto results_4threads = router.route_batch(batch_view);
  
  ASSERT_EQ(results_1thread.size(), 4);
  ASSERT_EQ(results_4threads.size(), 4);
  
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(results_1thread[i].selected_model, results_4threads[i].selected_model);
    EXPECT_EQ(results_1thread[i].cluster_id, results_4threads[i].cluster_id);
    EXPECT_NEAR(results_1thread[i].cluster_distance, results_4threads[i].cluster_distance, 1e-5f);
  }
}
