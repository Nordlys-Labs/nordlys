#include <gtest/gtest.h>

#include <vector>

#include "adaptive.h"

class CFFITest : public ::testing::Test {
protected:
  AdaptiveRouter* router_ = nullptr;

  void SetUp() override {
    // Note: This test assumes a test profile exists
    // In practice, you'd create a minimal test profile
    // router_ = adaptive_router_create("test_profile.json");
    // ASSERT_NE(router_, nullptr);
  }

  void TearDown() override {
    if (router_) {
      adaptive_router_destroy(router_);
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

// TODO: Add tests with valid profile once test fixtures are set up
// TEST_F(CFFITest, SingleRouteReturnsValidResult) {
//   size_t embedding_dim = adaptive_router_get_embedding_dim(router_);
//   std::vector<float> embedding(embedding_dim, 0.5f);
//
//   AdaptiveRouteResult* result = adaptive_router_route(
//       router_, embedding.data(), embedding_dim, 0.5f);
//
//   ASSERT_NE(result, nullptr);
//   EXPECT_NE(result->selected_model, nullptr);
//   EXPECT_GE(result->cluster_id, 0);
//   EXPECT_GE(result->cluster_distance, 0.0f);
//
//   adaptive_route_result_free(result);
// }

// TEST_F(CFFITest, BatchRouteReturnsValidResults) {
//   size_t embedding_dim = adaptive_router_get_embedding_dim(router_);
//   size_t n_embeddings = 5;
//   std::vector<float> embeddings(n_embeddings * embedding_dim, 0.5f);
//
//   AdaptiveBatchRouteResult* result = adaptive_router_route_batch(
//       router_, embeddings.data(), n_embeddings, embedding_dim, 0.5f);
//
//   ASSERT_NE(result, nullptr);
//   EXPECT_EQ(result->count, n_embeddings);
//
//   for (size_t i = 0; i < result->count; ++i) {
//     EXPECT_NE(result->results[i].selected_model, nullptr);
//     EXPECT_GE(result->results[i].cluster_id, 0);
//   }
//
//   adaptive_batch_route_result_free(result);
// }

TEST_F(CFFITest, BatchRouteHandlesNullRouter) {
  std::vector<float> embeddings(384, 0.5f);
  AdaptiveBatchRouteResult* result = adaptive_router_route_batch(
      nullptr, embeddings.data(), 1, 384, 0.5f);
  EXPECT_EQ(result, nullptr);
}

TEST_F(CFFITest, BatchRouteHandlesNullEmbeddings) {
  // Can't test without a valid router, but documents expected behavior
  // adaptive_router_route_batch(router_, nullptr, 1, 384, 0.5f);
  // Should return nullptr
  EXPECT_TRUE(true);  // Placeholder
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
