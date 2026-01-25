#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "nordlys.h"

// Test profile JSON for creating valid nordlys instances (v2.0 format)
static const char* kTestProfileJson = R"({
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

class CFFITest : public ::testing::Test {
protected:
  Nordlys* nordlys_ = nullptr;

  void SetUp() override {
    // Create nordlys from JSON string for testing
    nordlys_ = nordlys_create_from_json(kTestProfileJson, NORDLYS_DEVICE_CPU);
    ASSERT_NE(nordlys_, nullptr) << "Failed to create nordlys from JSON";
  }

  void TearDown() override {
    if (nordlys_) {
      nordlys_destroy(nordlys_);
      nordlys_ = nullptr;
    }
  }
};

TEST_F(CFFITest, CreationFailsWithInvalidPath) {
  Nordlys* nordlys = nordlys_create("nonexistent_file.json", NORDLYS_DEVICE_CPU);
  EXPECT_EQ(nordlys, nullptr);
}

TEST_F(CFFITest, CreationFromJsonStringFailsWithInvalidJson) {
  Nordlys* nordlys = nordlys_create_from_json("invalid json", NORDLYS_DEVICE_CPU);
  EXPECT_EQ(nordlys, nullptr);
}

TEST_F(CFFITest, CreationFromBinaryFailsWithInvalidPath) {
  Nordlys* nordlys = nordlys_create_from_msgpack("nonexistent_file.msgpack", NORDLYS_DEVICE_CPU);
  EXPECT_EQ(nordlys, nullptr);
}

TEST_F(CFFITest, SingleRouteReturnsValidResult) {
  size_t embedding_dim = nordlys_get_embedding_dim(nordlys_);
  ASSERT_EQ(embedding_dim, 4);

  // Create embedding close to first cluster center [1,0,0,0]
  std::vector<float> embedding = {0.95f, 0.05f, 0.0f, 0.0f};

  NordlysErrorCode error;
  NordlysRouteResult* result = nordlys_route(nordlys_, embedding.data(), embedding_dim, &error);

  ASSERT_NE(result, nullptr);
  EXPECT_NE(result->selected_model, nullptr);
  // Model should be one of the two available models
  bool is_valid_model = (std::string(result->selected_model) == "provider1/gpt-4"
                         || std::string(result->selected_model) == "provider2/llama");
  EXPECT_TRUE(is_valid_model) << "Unexpected model: " << result->selected_model;
  EXPECT_EQ(result->cluster_id, 0);
  EXPECT_GE(result->cluster_distance, 0.0f);
  EXPECT_LT(result->cluster_distance, 1.0f);

  nordlys_route_result_free(result);
}

TEST_F(CFFITest, BatchRouteReturnsValidResults) {
  size_t embedding_dim = nordlys_get_embedding_dim(nordlys_);
  size_t n_embeddings = 3;

  // Create 3 embeddings, each close to a different cluster
  std::vector<float> embeddings = {// Embedding 1: close to cluster 0 [1,0,0,0]
                                   0.95f, 0.05f, 0.0f, 0.0f,
                                   // Embedding 2: close to cluster 1 [0,1,0,0]
                                   0.05f, 0.95f, 0.0f, 0.0f,
                                   // Embedding 3: close to cluster 2 [0,0,1,0]
                                   0.0f, 0.05f, 0.95f, 0.0f};

  NordlysErrorCode error;
  NordlysBatchRouteResult* result
      = nordlys_route_batch(nordlys_, embeddings.data(), n_embeddings, embedding_dim, &error);

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

  nordlys_batch_route_result_free(result);
}

TEST_F(CFFITest, GetNClustersReturnsCorrectValue) {
  size_t n_clusters = nordlys_get_n_clusters(nordlys_);
  EXPECT_EQ(n_clusters, 3);
}

TEST_F(CFFITest, GetEmbeddingDimReturnsCorrectValue) {
  size_t embedding_dim = nordlys_get_embedding_dim(nordlys_);
  EXPECT_EQ(embedding_dim, 4);
}

TEST_F(CFFITest, GetSupportedModelsReturnsModelList) {
  size_t count = 0;
  char** models = nordlys_get_supported_models(nordlys_, &count);

  ASSERT_NE(models, nullptr);
  EXPECT_EQ(count, 2);

  EXPECT_STREQ(models[0], "provider1/gpt-4");
  EXPECT_STREQ(models[1], "provider2/llama");

  nordlys_string_array_free(models, count);
}

TEST_F(CFFITest, RouteWithDifferentCostBiasSelectsDifferentModels) {
  size_t embedding_dim = nordlys_get_embedding_dim(nordlys_);
  std::vector<float> embedding = {0.95f, 0.05f, 0.0f, 0.0f};

  // Route with accuracy preference (cost_bias = 0.0)
  NordlysErrorCode error_accuracy;
  NordlysRouteResult* result_accuracy
      = nordlys_route(nordlys_, embedding.data(), embedding_dim, &error_accuracy);

  // Route with cost preference (cost_bias = 1.0)
  NordlysErrorCode error_cost;
  NordlysRouteResult* result_cost
      = nordlys_route(nordlys_, embedding.data(), embedding_dim, &error_cost);

  ASSERT_NE(result_accuracy, nullptr);
  ASSERT_NE(result_cost, nullptr);

  // Both should assign to same cluster
  EXPECT_EQ(result_accuracy->cluster_id, result_cost->cluster_id);

  // Models might differ based on cost optimization
  EXPECT_NE(result_accuracy->selected_model, nullptr);
  EXPECT_NE(result_cost->selected_model, nullptr);

  nordlys_route_result_free(result_accuracy);
  nordlys_route_result_free(result_cost);
}

TEST_F(CFFITest, RouteWithWrongDimensionReturnsNull) {
  // Try routing with wrong embedding dimension (3 instead of 4)
  std::vector<float> wrong_embedding = {1.0f, 0.0f, 0.0f};

  NordlysErrorCode error;
  NordlysRouteResult* result
      = nordlys_route(nordlys_, wrong_embedding.data(), wrong_embedding.size(), &error);

  EXPECT_EQ(result, nullptr);
}

TEST_F(CFFITest, RouteResultHasAlternatives) {
  size_t embedding_dim = nordlys_get_embedding_dim(nordlys_);
  std::vector<float> embedding = {0.95f, 0.05f, 0.0f, 0.0f};

  NordlysErrorCode error;
  NordlysRouteResult* result = nordlys_route(nordlys_, embedding.data(), embedding_dim, &error);

  ASSERT_NE(result, nullptr);

  // Should have at least 1 alternative (we have 2 models, max_alternatives=2)
  EXPECT_GE(result->alternatives_count, 1);
  EXPECT_LE(result->alternatives_count, 2);

  if (result->alternatives_count > 0) {
    EXPECT_NE(result->alternatives, nullptr);
    EXPECT_NE(result->alternatives[0], nullptr);
  }

  nordlys_route_result_free(result);
}

TEST_F(CFFITest, BatchRouteHandlesNullNordlys) {
  std::vector<float> embeddings(384, 0.5f);
  NordlysErrorCode error;
  NordlysBatchRouteResult* result
      = nordlys_route_batch(nullptr, embeddings.data(), 1, 384, &error);
  EXPECT_EQ(result, nullptr);
}

TEST_F(CFFITest, BatchRouteHandlesNullEmbeddings) {
  size_t embedding_dim = nordlys_get_embedding_dim(nordlys_);

  // Pass nullptr embeddings - should return nullptr
  NordlysErrorCode error;
  NordlysBatchRouteResult* result
      = nordlys_route_batch(nordlys_, nullptr, 1, embedding_dim, &error);
  EXPECT_EQ(result, nullptr);
}

TEST_F(CFFITest, RouteHandlesNullNordlys) {
  std::vector<float> embedding = {1.0f, 0.0f, 0.0f, 0.0f};

  NordlysErrorCode error;
  NordlysRouteResult* result
      = nordlys_route(nullptr, embedding.data(), embedding.size(), &error);
  EXPECT_EQ(result, nullptr);
}

TEST_F(CFFITest, GetNClustersHandlesNullNordlys) {
  size_t n_clusters = nordlys_get_n_clusters(nullptr);
  EXPECT_EQ(n_clusters, 0);
}

TEST_F(CFFITest, GetEmbeddingDimHandlesNullNordlys) {
  size_t embedding_dim = nordlys_get_embedding_dim(nullptr);
  EXPECT_EQ(embedding_dim, 0);
}

TEST_F(CFFITest, GetSupportedModelsHandlesNullNordlys) {
  size_t count = 99;  // Initialize to non-zero to verify it's set to 0
  char** models = nordlys_get_supported_models(nullptr, &count);
  EXPECT_EQ(models, nullptr);
  EXPECT_EQ(count, 0);
}

TEST_F(CFFITest, StringFreeHandlesNull) {
  // Should not crash
  nordlys_string_free(nullptr);
  EXPECT_TRUE(true);
}

TEST_F(CFFITest, RouteResultFreeHandlesNull) {
  // Should not crash
  nordlys_route_result_free(nullptr);
  EXPECT_TRUE(true);
}

TEST_F(CFFITest, BatchRouteResultFreeHandlesNull) {
  // Should not crash
  nordlys_batch_route_result_free(nullptr);
  EXPECT_TRUE(true);
}

TEST_F(CFFITest, DestroyHandlesNull) {
  // Should not crash
  nordlys_destroy(nullptr);
  EXPECT_TRUE(true);
}

// Error code verification tests
TEST_F(CFFITest, RouteSetsErrorCodes) {
  std::vector<float> embedding = {0.95f, 0.05f, 0.0f, 0.0f};
  NordlysErrorCode error;

  // Valid call should succeed
  NordlysRouteResult* result
      = nordlys_route(nordlys_, embedding.data(), embedding.size(), &error);
  EXPECT_EQ(error, NORDLYS_OK);
  EXPECT_NE(result, nullptr);
  nordlys_route_result_free(result);

  // Null nordlys should set error
  result = nordlys_route(nullptr, embedding.data(), embedding.size(), &error);
  EXPECT_EQ(error, NORDLYS_ERROR_NULL_NORDLYS);
  EXPECT_EQ(result, nullptr);

  // Null embedding should set error
  result = nordlys_route(nordlys_, nullptr, embedding.size(), &error);
  EXPECT_EQ(error, NORDLYS_ERROR_NULL_EMBEDDING);
  EXPECT_EQ(result, nullptr);
}
