#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "adaptive.h"

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

class CFFITest : public ::testing::Test {
protected:
  AdaptiveRouter* router_ = nullptr;

  void SetUp() override {
    // Create router from JSON string for testing
    router_ = adaptive_router_create_from_json(kTestProfileJson);
    ASSERT_NE(router_, nullptr) << "Failed to create router from JSON";
  }

  void TearDown() override {
    if (router_) {
      adaptive_router_destroy(router_);
      router_ = nullptr;
    }
  }
};

TEST_F(CFFITest, RouterCreationFailsWithInvalidPath) {
  AdaptiveRouter* router = adaptive_router_create("nonexistent_file.json");
  EXPECT_EQ(router, nullptr);
}

TEST_F(CFFITest, RouterCreationFromJsonStringFailsWithInvalidJson) {
  AdaptiveRouter* router = adaptive_router_create_from_json("invalid json");
  EXPECT_EQ(router, nullptr);
}

TEST_F(CFFITest, RouterCreationFromBinaryFailsWithInvalidPath) {
  AdaptiveRouter* router = adaptive_router_create_from_binary("nonexistent_file.msgpack");
  EXPECT_EQ(router, nullptr);
}

TEST_F(CFFITest, SingleRouteReturnsValidResult) {
  size_t embedding_dim = adaptive_router_get_embedding_dim(router_);
  ASSERT_EQ(embedding_dim, 4);

  // Create embedding close to first cluster center [1,0,0,0]
  std::vector<float> embedding = {0.95f, 0.05f, 0.0f, 0.0f};

  AdaptiveRouteResult* result = adaptive_router_route(
      router_, embedding.data(), embedding_dim, 0.5f);

  ASSERT_NE(result, nullptr);
  EXPECT_NE(result->selected_model, nullptr);
  // Model should be one of the two available models
  bool is_valid_model = (std::string(result->selected_model) == "provider1/gpt-4" ||
                         std::string(result->selected_model) == "provider2/llama");
  EXPECT_TRUE(is_valid_model) << "Unexpected model: " << result->selected_model;
  EXPECT_EQ(result->cluster_id, 0);
  EXPECT_GE(result->cluster_distance, 0.0f);
  EXPECT_LT(result->cluster_distance, 1.0f);

  adaptive_route_result_free(result);
}

TEST_F(CFFITest, SingleRouteWithDoubleReturnsValidResult) {
  size_t embedding_dim = adaptive_router_get_embedding_dim(router_);
  ASSERT_EQ(embedding_dim, 4);

  // Create double-precision embedding close to second cluster [0,1,0,0]
  std::vector<double> embedding = {0.05, 0.95, 0.0, 0.0};

  AdaptiveRouteResult* result = adaptive_router_route_double(
      router_, embedding.data(), embedding_dim, 0.5f);

  ASSERT_NE(result, nullptr);
  EXPECT_NE(result->selected_model, nullptr);
  EXPECT_EQ(result->cluster_id, 1);
  EXPECT_GE(result->cluster_distance, 0.0f);

  adaptive_route_result_free(result);
}

TEST_F(CFFITest, RouteSimpleReturnsModelId) {
  size_t embedding_dim = adaptive_router_get_embedding_dim(router_);
  std::vector<float> embedding = {0.0f, 0.0f, 0.95f, 0.05f};

  char* model_id = adaptive_router_route_simple(
      router_, embedding.data(), embedding_dim, 0.5f);

  ASSERT_NE(model_id, nullptr);
  // Verify it returns a valid model
  bool is_valid_model = (std::string(model_id) == "provider1/gpt-4" ||
                         std::string(model_id) == "provider2/llama");
  EXPECT_TRUE(is_valid_model) << "Unexpected model: " << model_id;

  adaptive_string_free(model_id);
}

TEST_F(CFFITest, BatchRouteReturnsValidResults) {
  size_t embedding_dim = adaptive_router_get_embedding_dim(router_);
  size_t n_embeddings = 3;

  // Create 3 embeddings, each close to a different cluster
  std::vector<float> embeddings = {
    // Embedding 1: close to cluster 0 [1,0,0,0]
    0.95f, 0.05f, 0.0f, 0.0f,
    // Embedding 2: close to cluster 1 [0,1,0,0]
    0.05f, 0.95f, 0.0f, 0.0f,
    // Embedding 3: close to cluster 2 [0,0,1,0]
    0.0f, 0.05f, 0.95f, 0.0f
  };

  AdaptiveBatchRouteResult* result = adaptive_router_route_batch(
      router_, embeddings.data(), n_embeddings, embedding_dim, 0.5f);

  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->count, n_embeddings);

  for (size_t i = 0; i < result->count; ++i) {
    EXPECT_NE(result->results[i].selected_model, nullptr);
    EXPECT_GE(result->results[i].cluster_id, 0);
    EXPECT_LT(result->results[i].cluster_id, 3);
    EXPECT_GE(result->results[i].cluster_distance, 0.0f);
  }

  // Verify cluster assignments match expected
  EXPECT_EQ(result->results[0].cluster_id, 0);
  EXPECT_EQ(result->results[1].cluster_id, 1);
  EXPECT_EQ(result->results[2].cluster_id, 2);

  adaptive_batch_route_result_free(result);
}

TEST_F(CFFITest, BatchRouteWithDoubleReturnsValidResults) {
  size_t embedding_dim = adaptive_router_get_embedding_dim(router_);
  size_t n_embeddings = 2;

  std::vector<double> embeddings = {
    0.95, 0.05, 0.0, 0.0,
    0.0, 0.0, 0.95, 0.05
  };

  AdaptiveBatchRouteResult* result = adaptive_router_route_batch_double(
      router_, embeddings.data(), n_embeddings, embedding_dim, 0.5f);

  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->count, n_embeddings);

  adaptive_batch_route_result_free(result);
}

TEST_F(CFFITest, GetNClustersReturnsCorrectValue) {
  size_t n_clusters = adaptive_router_get_n_clusters(router_);
  EXPECT_EQ(n_clusters, 3);
}

TEST_F(CFFITest, GetEmbeddingDimReturnsCorrectValue) {
  size_t embedding_dim = adaptive_router_get_embedding_dim(router_);
  EXPECT_EQ(embedding_dim, 4);
}

TEST_F(CFFITest, GetSupportedModelsReturnsModelList) {
  size_t count = 0;
  char** models = adaptive_router_get_supported_models(router_, &count);

  ASSERT_NE(models, nullptr);
  EXPECT_EQ(count, 2);

  EXPECT_STREQ(models[0], "provider1/gpt-4");
  EXPECT_STREQ(models[1], "provider2/llama");

  adaptive_string_array_free(models, count);
}

TEST_F(CFFITest, RouteWithDifferentCostBiasSelectsDifferentModels) {
  size_t embedding_dim = adaptive_router_get_embedding_dim(router_);
  std::vector<float> embedding = {0.95f, 0.05f, 0.0f, 0.0f};

  // Route with accuracy preference (cost_bias = 0.0)
  AdaptiveRouteResult* result_accuracy = adaptive_router_route(
      router_, embedding.data(), embedding_dim, 0.0f);

  // Route with cost preference (cost_bias = 1.0)
  AdaptiveRouteResult* result_cost = adaptive_router_route(
      router_, embedding.data(), embedding_dim, 1.0f);

  ASSERT_NE(result_accuracy, nullptr);
  ASSERT_NE(result_cost, nullptr);

  // Both should assign to same cluster
  EXPECT_EQ(result_accuracy->cluster_id, result_cost->cluster_id);

  // Models might differ based on cost optimization
  EXPECT_NE(result_accuracy->selected_model, nullptr);
  EXPECT_NE(result_cost->selected_model, nullptr);

  adaptive_route_result_free(result_accuracy);
  adaptive_route_result_free(result_cost);
}

TEST_F(CFFITest, RouteWithWrongDimensionReturnsNull) {
  // Try routing with wrong embedding dimension (3 instead of 4)
  std::vector<float> wrong_embedding = {1.0f, 0.0f, 0.0f};

  AdaptiveRouteResult* result = adaptive_router_route(
      router_, wrong_embedding.data(), wrong_embedding.size(), 0.5f);

  EXPECT_EQ(result, nullptr);
}

TEST_F(CFFITest, RouteResultHasAlternatives) {
  size_t embedding_dim = adaptive_router_get_embedding_dim(router_);
  std::vector<float> embedding = {0.95f, 0.05f, 0.0f, 0.0f};

  AdaptiveRouteResult* result = adaptive_router_route(
      router_, embedding.data(), embedding_dim, 0.5f);

  ASSERT_NE(result, nullptr);

  // Should have at least 1 alternative (we have 2 models, max_alternatives=2)
  EXPECT_GE(result->alternatives_count, 1);
  EXPECT_LE(result->alternatives_count, 2);

  if (result->alternatives_count > 0) {
    EXPECT_NE(result->alternatives, nullptr);
    EXPECT_NE(result->alternatives[0], nullptr);
  }

  adaptive_route_result_free(result);
}

TEST_F(CFFITest, BatchRouteHandlesNullRouter) {
  std::vector<float> embeddings(384, 0.5f);
  AdaptiveBatchRouteResult* result = adaptive_router_route_batch(
      nullptr, embeddings.data(), 1, 384, 0.5f);
  EXPECT_EQ(result, nullptr);
}

TEST_F(CFFITest, BatchRouteHandlesNullEmbeddings) {
  size_t embedding_dim = adaptive_router_get_embedding_dim(router_);

  // Pass nullptr embeddings - should return nullptr
  AdaptiveBatchRouteResult* result = adaptive_router_route_batch(
      router_, nullptr, 1, embedding_dim, 0.5f);
  EXPECT_EQ(result, nullptr);
}

TEST_F(CFFITest, RouteHandlesNullRouter) {
  std::vector<float> embedding = {1.0f, 0.0f, 0.0f, 0.0f};

  AdaptiveRouteResult* result = adaptive_router_route(
      nullptr, embedding.data(), embedding.size(), 0.5f);
  EXPECT_EQ(result, nullptr);
}

TEST_F(CFFITest, RouteSimpleHandlesNullRouter) {
  std::vector<float> embedding = {1.0f, 0.0f, 0.0f, 0.0f};

  char* model_id = adaptive_router_route_simple(
      nullptr, embedding.data(), embedding.size(), 0.5f);
  EXPECT_EQ(model_id, nullptr);
}

TEST_F(CFFITest, GetNClustersHandlesNullRouter) {
  size_t n_clusters = adaptive_router_get_n_clusters(nullptr);
  EXPECT_EQ(n_clusters, 0);
}

TEST_F(CFFITest, GetEmbeddingDimHandlesNullRouter) {
  size_t embedding_dim = adaptive_router_get_embedding_dim(nullptr);
  EXPECT_EQ(embedding_dim, 0);
}

TEST_F(CFFITest, GetSupportedModelsHandlesNullRouter) {
  size_t count = 99;  // Initialize to non-zero to verify it's set to 0
  char** models = adaptive_router_get_supported_models(nullptr, &count);
  EXPECT_EQ(models, nullptr);
  EXPECT_EQ(count, 0);
}

TEST_F(CFFITest, StringFreeHandlesNull) {
  // Should not crash
  adaptive_string_free(nullptr);
  EXPECT_TRUE(true);
}

TEST_F(CFFITest, RouteResultFreeHandlesNull) {
  // Should not crash
  adaptive_route_result_free(nullptr);
  EXPECT_TRUE(true);
}

TEST_F(CFFITest, BatchRouteResultFreeHandlesNull) {
  // Should not crash
  adaptive_batch_route_result_free(nullptr);
  EXPECT_TRUE(true);
}

TEST_F(CFFITest, RouterDestroyHandlesNull) {
  // Should not crash
  adaptive_router_destroy(nullptr);
  EXPECT_TRUE(true);
}
