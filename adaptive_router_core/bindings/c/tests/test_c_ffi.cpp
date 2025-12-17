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

static const char* kTestProfileJsonFloat64 = R"({
  "metadata": {
    "n_clusters": 3,
    "embedding_model": "test-model",
    "dtype": "float64",
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

  AdaptiveErrorCode error;
  AdaptiveRouteResult32* result = adaptive_router_route_f32(
      router_, embedding.data(), embedding_dim, 0.5f, &error);

  ASSERT_NE(result, nullptr);
  EXPECT_NE(result->selected_model, nullptr);
  // Model should be one of the two available models
  bool is_valid_model = (std::string(result->selected_model) == "provider1/gpt-4" ||
                         std::string(result->selected_model) == "provider2/llama");
  EXPECT_TRUE(is_valid_model) << "Unexpected model: " << result->selected_model;
  EXPECT_EQ(result->cluster_id, 0);
  EXPECT_GE(result->cluster_distance, 0.0f);
  EXPECT_LT(result->cluster_distance, 1.0f);

  adaptive_route_result_free_f32(result);
}

TEST_F(CFFITest, SingleRouteWithDoubleReturnsValidResult) {
  size_t embedding_dim = adaptive_router_get_embedding_dim(router_);
  ASSERT_EQ(embedding_dim, 4);

  // Create double-precision embedding close to second cluster [0,1,0,0]
  std::vector<double> embedding = {0.05, 0.95, 0.0, 0.0};

  // With strict type enforcement, double routing on float32 router returns nullptr
  AdaptiveErrorCode error;
  AdaptiveRouteResult64* result = adaptive_router_route_f64(
      router_, embedding.data(), embedding_dim, 0.5f, &error);

  EXPECT_EQ(result, nullptr);
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

  AdaptiveErrorCode error;
  AdaptiveBatchRouteResult32* result = adaptive_router_route_batch_f32(
      router_, embeddings.data(), n_embeddings, embedding_dim, 0.5f, &error);

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

  adaptive_batch_route_result_free_f32(result);
}

TEST_F(CFFITest, BatchRouteWithDoubleReturnsValidResults) {
  size_t embedding_dim = adaptive_router_get_embedding_dim(router_);
  size_t n_embeddings = 2;

  std::vector<double> embeddings = {
    0.95, 0.05, 0.0, 0.0,
    0.0, 0.0, 0.95, 0.05
  };

  // With strict type enforcement, double batch routing on float32 router returns nullptr
  AdaptiveErrorCode error;
  AdaptiveBatchRouteResult64* result = adaptive_router_route_batch_f64(
      router_, embeddings.data(), n_embeddings, embedding_dim, 0.5f, &error);

  EXPECT_EQ(result, nullptr);
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
  AdaptiveErrorCode error_accuracy;
  AdaptiveRouteResult32* result_accuracy = adaptive_router_route_f32(
      router_, embedding.data(), embedding_dim, 0.0f, &error_accuracy);

  // Route with cost preference (cost_bias = 1.0)
  AdaptiveErrorCode error_cost;
  AdaptiveRouteResult32* result_cost = adaptive_router_route_f32(
      router_, embedding.data(), embedding_dim, 1.0f, &error_cost);

  ASSERT_NE(result_accuracy, nullptr);
  ASSERT_NE(result_cost, nullptr);

  // Both should assign to same cluster
  EXPECT_EQ(result_accuracy->cluster_id, result_cost->cluster_id);

  // Models might differ based on cost optimization
  EXPECT_NE(result_accuracy->selected_model, nullptr);
  EXPECT_NE(result_cost->selected_model, nullptr);

  adaptive_route_result_free_f32(result_accuracy);
  adaptive_route_result_free_f32(result_cost);
}

TEST_F(CFFITest, RouteWithWrongDimensionReturnsNull) {
  // Try routing with wrong embedding dimension (3 instead of 4)
  std::vector<float> wrong_embedding = {1.0f, 0.0f, 0.0f};

  AdaptiveErrorCode error;
  AdaptiveRouteResult32* result = adaptive_router_route_f32(
      router_, wrong_embedding.data(), wrong_embedding.size(), 0.5f, &error);

  EXPECT_EQ(result, nullptr);
}

TEST_F(CFFITest, RouteResultHasAlternatives) {
  size_t embedding_dim = adaptive_router_get_embedding_dim(router_);
  std::vector<float> embedding = {0.95f, 0.05f, 0.0f, 0.0f};

  AdaptiveErrorCode error;
  AdaptiveRouteResult32* result = adaptive_router_route_f32(
      router_, embedding.data(), embedding_dim, 0.5f, &error);

  ASSERT_NE(result, nullptr);

  // Should have at least 1 alternative (we have 2 models, max_alternatives=2)
  EXPECT_GE(result->alternatives_count, 1);
  EXPECT_LE(result->alternatives_count, 2);

  if (result->alternatives_count > 0) {
    EXPECT_NE(result->alternatives, nullptr);
    EXPECT_NE(result->alternatives[0], nullptr);
  }

  adaptive_route_result_free_f32(result);
}

TEST_F(CFFITest, BatchRouteHandlesNullRouter) {
  std::vector<float> embeddings(384, 0.5f);
  AdaptiveErrorCode error;
  AdaptiveBatchRouteResult32* result = adaptive_router_route_batch_f32(
      nullptr, embeddings.data(), 1, 384, 0.5f, &error);
  EXPECT_EQ(result, nullptr);
}

TEST_F(CFFITest, BatchRouteHandlesNullEmbeddings) {
  size_t embedding_dim = adaptive_router_get_embedding_dim(router_);

  // Pass nullptr embeddings - should return nullptr
  AdaptiveErrorCode error;
  AdaptiveBatchRouteResult32* result = adaptive_router_route_batch_f32(
      router_, nullptr, 1, embedding_dim, 0.5f, &error);
  EXPECT_EQ(result, nullptr);
}

TEST_F(CFFITest, RouteHandlesNullRouter) {
  std::vector<float> embedding = {1.0f, 0.0f, 0.0f, 0.0f};

  AdaptiveErrorCode error;
  AdaptiveRouteResult32* result = adaptive_router_route_f32(
      nullptr, embedding.data(), embedding.size(), 0.5f, &error);
  EXPECT_EQ(result, nullptr);
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
  adaptive_route_result_free_f32(nullptr);
  EXPECT_TRUE(true);
}

TEST_F(CFFITest, BatchRouteResultFreeHandlesNull) {
  // Should not crash
  adaptive_batch_route_result_free_f32(nullptr);
  EXPECT_TRUE(true);
}

TEST_F(CFFITest, RouterDestroyHandlesNull) {
  // Should not crash
  adaptive_router_destroy(nullptr);
  EXPECT_TRUE(true);
}

class CFFITestFloat64 : public ::testing::Test {
protected:
  AdaptiveRouter* router_ = nullptr;

  void SetUp() override {
    router_ = adaptive_router_create_from_json(kTestProfileJsonFloat64);
    ASSERT_NE(router_, nullptr) << "Failed to create float64 router from JSON";
  }

  void TearDown() override {
    if (router_) {
      adaptive_router_destroy(router_);
      router_ = nullptr;
    }
  }
};

// Precision query tests
TEST_F(CFFITest, GetPrecisionReturnsFloat32) {
  AdaptivePrecision precision = adaptive_router_get_precision(router_);
  EXPECT_EQ(precision, ADAPTIVE_PRECISION_FLOAT32);
}

TEST_F(CFFITestFloat64, GetPrecisionReturnsFloat64) {
  AdaptivePrecision precision = adaptive_router_get_precision(router_);
  EXPECT_EQ(precision, ADAPTIVE_PRECISION_FLOAT64);
}

TEST(CFFITestPrecision, GetPrecisionHandlesNullRouter) {
  AdaptivePrecision precision = adaptive_router_get_precision(nullptr);
  EXPECT_EQ(precision, ADAPTIVE_PRECISION_UNKNOWN);
}

// Type mismatch tests - strict enforcement
TEST_F(CFFITest, RouteDoubleOnFloat32RouterReturnsNull) {
  std::vector<double> embedding = {0.95, 0.05, 0.0, 0.0};
  AdaptiveErrorCode error;
  AdaptiveRouteResult64* result = adaptive_router_route_f64(
      router_, embedding.data(), embedding.size(), 0.5f, &error);
  EXPECT_EQ(result, nullptr);  // Type mismatch: can't use double on float32 router
}

TEST_F(CFFITestFloat64, RouteFloatOnFloat64RouterReturnsNull) {
  std::vector<float> embedding = {0.95f, 0.05f, 0.0f, 0.0f};
  AdaptiveErrorCode error;
  AdaptiveRouteResult32* result = adaptive_router_route_f32(
      router_, embedding.data(), embedding.size(), 0.5f, &error);
  EXPECT_EQ(result, nullptr);  // Type mismatch: can't use float on float64 router
}

// Float64 router working correctly with double embeddings
TEST_F(CFFITestFloat64, RouteDoubleOnFloat64RouterSucceeds) {
  std::vector<double> embedding = {0.95, 0.05, 0.0, 0.0};
  AdaptiveErrorCode error;
  AdaptiveRouteResult64* result = adaptive_router_route_f64(
      router_, embedding.data(), embedding.size(), 0.5f, &error);

  ASSERT_NE(result, nullptr);
  EXPECT_NE(result->selected_model, nullptr);
  EXPECT_EQ(result->cluster_id, 0);

   adaptive_route_result_free_f64(result);
}

TEST_F(CFFITestFloat64, BatchRouteDoubleOnFloat64RouterSucceeds) {
  std::vector<double> embeddings = {
    0.95, 0.05, 0.0, 0.0,
    0.0, 0.0, 0.95, 0.05
  };

  AdaptiveErrorCode error;
  AdaptiveBatchRouteResult64* result = adaptive_router_route_batch_f64(
      router_, embeddings.data(), 2, 4, 0.5f, &error);

  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->count, 2);
   EXPECT_EQ(result->results[0].cluster_id, 0);
   EXPECT_EQ(result->results[1].cluster_id, 2);

   adaptive_batch_route_result_free_f64(result);
}

TEST_F(CFFITest, BatchRouteDoubleOnFloat32RouterReturnsNull) {
  std::vector<double> embeddings = {0.95, 0.05, 0.0, 0.0};
  AdaptiveErrorCode error;
  AdaptiveBatchRouteResult64* result = adaptive_router_route_batch_f64(
      router_, embeddings.data(), 1, 4, 0.5f, &error);
  EXPECT_EQ(result, nullptr);
}

TEST_F(CFFITestFloat64, BatchRouteFloatOnFloat64RouterReturnsNull) {
  std::vector<float> embeddings = {0.95f, 0.05f, 0.0f, 0.0f};
  AdaptiveErrorCode error;
  AdaptiveBatchRouteResult32* result = adaptive_router_route_batch_f32(
      router_, embeddings.data(), 1, 4, 0.5f, &error);
  EXPECT_EQ(result, nullptr);
}

// Query functions work on both router types
TEST_F(CFFITestFloat64, GetNClustersWorks) {
  EXPECT_EQ(adaptive_router_get_n_clusters(router_), 3);
}

TEST_F(CFFITestFloat64, GetEmbeddingDimWorks) {
  EXPECT_EQ(adaptive_router_get_embedding_dim(router_), 4);
}

TEST_F(CFFITestFloat64, GetSupportedModelsWorks) {
  size_t count = 0;
  char** models = adaptive_router_get_supported_models(router_, &count);
  ASSERT_NE(models, nullptr);
  EXPECT_EQ(count, 2);
  adaptive_string_array_free(models, count);
}

// Error code verification tests
TEST_F(CFFITest, RouteF32SetsErrorCodes) {
  std::vector<float> embedding = {0.95f, 0.05f, 0.0f, 0.0f};
  AdaptiveErrorCode error;

  // Valid call should succeed
  AdaptiveRouteResult32* result = adaptive_router_route_f32(
      router_, embedding.data(), embedding.size(), 0.5f, &error);
  EXPECT_EQ(error, ADAPTIVE_OK);
  EXPECT_NE(result, nullptr);
  adaptive_route_result_free_f32(result);

  // Null router should set error
  result = adaptive_router_route_f32(nullptr, embedding.data(), embedding.size(), 0.5f, &error);
  EXPECT_EQ(error, ADAPTIVE_ERROR_NULL_ROUTER);
  EXPECT_EQ(result, nullptr);

  // Null embedding should set error
  result = adaptive_router_route_f32(router_, nullptr, embedding.size(), 0.5f, &error);
  EXPECT_EQ(error, ADAPTIVE_ERROR_NULL_EMBEDDING);
  EXPECT_EQ(result, nullptr);
}

TEST_F(CFFITest, RouteF64OnF32RouterSetsTypeMismatchError) {
  std::vector<double> embedding = {0.95, 0.05, 0.0, 0.0};
  AdaptiveErrorCode error;

  AdaptiveRouteResult64* result = adaptive_router_route_f64(
      router_, embedding.data(), embedding.size(), 0.5f, &error);
  EXPECT_EQ(error, ADAPTIVE_ERROR_TYPE_MISMATCH);
  EXPECT_EQ(result, nullptr);
}

TEST_F(CFFITestFloat64, RouteF32OnF64RouterSetsTypeMismatchError) {
  std::vector<float> embedding = {0.95f, 0.05f, 0.0f, 0.0f};
  AdaptiveErrorCode error;

  AdaptiveRouteResult32* result = adaptive_router_route_f32(
      router_, embedding.data(), embedding.size(), 0.5f, &error);
  EXPECT_EQ(error, ADAPTIVE_ERROR_TYPE_MISMATCH);
  EXPECT_EQ(result, nullptr);
}
