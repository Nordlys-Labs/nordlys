#include <gtest/gtest.h>
#include <cstdio>
#include <random>
#include <adaptive_core/profile.hpp>

// Suppress nodiscard warnings in tests since EXPECT_THROW requires ignoring return values
#pragma GCC diagnostic ignored "-Wunused-result"

// Test profile JSON for float32 profiles
static const char* kTestProfileJson = R"({
  "metadata": {
    "n_clusters": 3,
    "embedding_model": "test-model",
    "dtype": "float32",
    "silhouette_score": 0.85,
    "clustering": {
      "max_iter": 300,
      "n_init": 10,
      "algorithm": "lloyd"
    },
    "routing": {
      "lambda_min": 0.0,
      "lambda_max": 2.0,
      "default_cost_preference": 0.5,
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
      "provider": "openai",
      "model_name": "gpt-4",
      "cost_per_1m_input_tokens": 30.0,
      "cost_per_1m_output_tokens": 60.0,
      "error_rates": [0.1, 0.05, 0.15]
    },
    {
      "provider": "anthropic",
      "model_name": "claude",
      "cost_per_1m_input_tokens": 15.0,
      "cost_per_1m_output_tokens": 45.0,
      "error_rates": [0.08, 0.12, 0.06]
    },
    {
      "provider": "google",
      "model_name": "gemini-pro",
      "cost_per_1m_input_tokens": 0.5,
      "cost_per_1m_output_tokens": 1.5,
      "error_rates": [0.2, 0.18, 0.25]
    }
  ]
})";

// Test profile JSON for float64 profiles
static const char* kTestProfileJsonFloat64 = R"({
  "metadata": {
    "n_clusters": 3,
    "embedding_model": "test-model",
    "dtype": "float64",
    "silhouette_score": 0.85,
    "clustering": {
      "max_iter": 300,
      "n_init": 10,
      "algorithm": "lloyd"
    },
    "routing": {
      "lambda_min": 0.0,
      "lambda_max": 2.0,
      "default_cost_preference": 0.5,
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
      "provider": "openai",
      "model_name": "gpt-4",
      "cost_per_1m_input_tokens": 30.0,
      "cost_per_1m_output_tokens": 60.0,
      "error_rates": [0.1, 0.05, 0.15]
    },
    {
      "provider": "anthropic",
      "model_name": "claude",
      "cost_per_1m_input_tokens": 15.0,
      "cost_per_1m_output_tokens": 45.0,
      "error_rates": [0.08, 0.12, 0.06]
    }
  ]
})";

class ProfileTest : public ::testing::Test {
protected:
  RouterProfile test_profile = RouterProfile::from_json_string(kTestProfileJson);
};

TEST_F(ProfileTest, RoundTripJson) {
  // Serialize to JSON
  std::string json_str = test_profile.to_json_string();

  // Deserialize from JSON
  RouterProfile loaded = RouterProfile::from_json_string(json_str);

  // Compare metadata
  EXPECT_EQ(loaded.metadata.n_clusters, test_profile.metadata.n_clusters);
  EXPECT_EQ(loaded.metadata.embedding_model, test_profile.metadata.embedding_model);
  EXPECT_EQ(loaded.metadata.dtype, test_profile.metadata.dtype);
  EXPECT_FLOAT_EQ(loaded.metadata.silhouette_score, test_profile.metadata.silhouette_score);

  // Compare clustering config
  EXPECT_EQ(loaded.metadata.clustering.max_iter, test_profile.metadata.clustering.max_iter);
  EXPECT_EQ(loaded.metadata.clustering.algorithm, test_profile.metadata.clustering.algorithm);

  // Compare routing config
  EXPECT_FLOAT_EQ(loaded.metadata.routing.lambda_min, test_profile.metadata.routing.lambda_min);
  EXPECT_FLOAT_EQ(loaded.metadata.routing.default_cost_preference, test_profile.metadata.routing.default_cost_preference);

  // Compare cluster centers
  std::visit([&](const auto& orig_centers) {
    std::visit([&](const auto& loaded_centers) {
      EXPECT_EQ(orig_centers.rows(), loaded_centers.rows());
      EXPECT_EQ(orig_centers.cols(), loaded_centers.cols());
      for (int i = 0; i < orig_centers.rows(); ++i) {
        for (int j = 0; j < orig_centers.cols(); ++j) {
          EXPECT_FLOAT_EQ(orig_centers(i, j), loaded_centers(i, j));
        }
      }
    }, loaded.cluster_centers);
  }, test_profile.cluster_centers);

  // Compare models
  EXPECT_EQ(loaded.models.size(), test_profile.models.size());
  for (size_t i = 0; i < loaded.models.size(); ++i) {
    EXPECT_EQ(loaded.models[i].provider, test_profile.models[i].provider);
    EXPECT_EQ(loaded.models[i].model_name, test_profile.models[i].model_name);
    EXPECT_FLOAT_EQ(loaded.models[i].cost_per_1m_input_tokens, test_profile.models[i].cost_per_1m_input_tokens);
    EXPECT_FLOAT_EQ(loaded.models[i].cost_per_1m_output_tokens, test_profile.models[i].cost_per_1m_output_tokens);
    EXPECT_EQ(loaded.models[i].error_rates.size(), test_profile.models[i].error_rates.size());
    for (size_t j = 0; j < loaded.models[i].error_rates.size(); ++j) {
      EXPECT_FLOAT_EQ(loaded.models[i].error_rates[j], test_profile.models[i].error_rates[j]);
    }
  }
}

TEST_F(ProfileTest, RoundTripMsgpack) {
  // Serialize to msgpack
  std::string msgpack_data = test_profile.to_binary_string();

  // Deserialize from msgpack
  RouterProfile loaded = RouterProfile::from_binary_string(msgpack_data);

  // Compare metadata
  EXPECT_EQ(loaded.metadata.n_clusters, test_profile.metadata.n_clusters);
  EXPECT_EQ(loaded.metadata.embedding_model, test_profile.metadata.embedding_model);
  EXPECT_EQ(loaded.metadata.dtype, test_profile.metadata.dtype);
  EXPECT_FLOAT_EQ(loaded.metadata.silhouette_score, test_profile.metadata.silhouette_score);

  // Compare cluster centers
  std::visit([&](const auto& orig_centers) {
    std::visit([&](const auto& loaded_centers) {
      EXPECT_EQ(orig_centers.rows(), loaded_centers.rows());
      EXPECT_EQ(orig_centers.cols(), loaded_centers.cols());
      for (int i = 0; i < orig_centers.rows(); ++i) {
        for (int j = 0; j < orig_centers.cols(); ++j) {
          EXPECT_FLOAT_EQ(orig_centers(i, j), loaded_centers(i, j));
        }
      }
    }, loaded.cluster_centers);
  }, test_profile.cluster_centers);

  // Compare models
  EXPECT_EQ(loaded.models.size(), test_profile.models.size());
  for (size_t i = 0; i < loaded.models.size(); ++i) {
    EXPECT_EQ(loaded.models[i].provider, test_profile.models[i].provider);
    EXPECT_EQ(loaded.models[i].model_name, test_profile.models[i].model_name);
    EXPECT_EQ(loaded.models[i].error_rates.size(), test_profile.models[i].error_rates.size());
  }
}

TEST_F(ProfileTest, FileOperations) {
  // Test JSON file operations
  std::string json_file = "/tmp/test_profile.json";
  test_profile.to_json(json_file);

  RouterProfile loaded_json = RouterProfile::from_json(json_file);

  EXPECT_EQ(loaded_json.metadata.n_clusters, test_profile.metadata.n_clusters);
  EXPECT_EQ(loaded_json.models.size(), test_profile.models.size());

  // Test msgpack file operations
  std::string msgpack_file = "/tmp/test_profile.msgpack";
  test_profile.to_binary(msgpack_file);

  RouterProfile loaded_msgpack = RouterProfile::from_binary(msgpack_file);

  EXPECT_EQ(loaded_msgpack.metadata.n_clusters, test_profile.metadata.n_clusters);
  EXPECT_EQ(loaded_msgpack.models.size(), test_profile.models.size());

  // Cleanup
  std::remove(json_file.c_str());
  std::remove(msgpack_file.c_str());
}

TEST_F(ProfileTest, Validation) {
  // Valid profile should pass validation
  EXPECT_NO_THROW(test_profile.validate());

  // Test invalid n_clusters
  RouterProfile invalid_profile = test_profile;
  invalid_profile.metadata.n_clusters = -1;
  EXPECT_THROW(invalid_profile.validate(), std::invalid_argument);

  // Test invalid dtype
  invalid_profile = test_profile;
  invalid_profile.metadata.dtype = "invalid";
  EXPECT_THROW(invalid_profile.validate(), std::invalid_argument);

  // Test mismatched error rates size
  invalid_profile = test_profile;
  invalid_profile.models[0].error_rates.resize(2);  // Should be 3
  EXPECT_THROW(invalid_profile.validate(), std::invalid_argument);

  // Test invalid error rate
  invalid_profile = test_profile;
  invalid_profile.models[0].error_rates[0] = 1.5f;  // Should be <= 1.0
  EXPECT_THROW(invalid_profile.validate(), std::invalid_argument);

  // Test negative cost
  invalid_profile = test_profile;
  invalid_profile.models[0].cost_per_1m_input_tokens = -1.0f;
  EXPECT_THROW(invalid_profile.validate(), std::invalid_argument);
}

TEST_F(ProfileTest, Float64Support) {
  // Load a float64 profile from JSON
  RouterProfile double_profile = RouterProfile::from_json_string(kTestProfileJsonFloat64);

  // Test that it's properly identified as float64
  EXPECT_EQ(double_profile.metadata.dtype, "float64");
  EXPECT_TRUE(double_profile.is_float64());
  EXPECT_FALSE(double_profile.is_float32());

  // Test JSON round trip
  std::string json_str = double_profile.to_json_string();
  RouterProfile loaded = RouterProfile::from_json_string(json_str);

  EXPECT_EQ(loaded.metadata.dtype, "float64");
  EXPECT_TRUE(loaded.is_float64());
  EXPECT_FALSE(loaded.is_float32());

  // Test msgpack round trip
  std::string msgpack_data = double_profile.to_binary_string();
  RouterProfile loaded_msgpack = RouterProfile::from_binary_string(msgpack_data);

  EXPECT_EQ(loaded_msgpack.metadata.dtype, "float64");
  EXPECT_TRUE(loaded_msgpack.is_float64());
}

TEST_F(ProfileTest, InvalidJsonParsing) {
  // Test invalid JSON string
  std::string invalid_json = "{ invalid json }";
  EXPECT_THROW(RouterProfile::from_json_string(invalid_json), std::exception);

  // Test missing required fields
  std::string missing_metadata = R"({
    "cluster_centers": {"n_clusters": 1, "feature_dim": 1, "cluster_centers": [[1.0]]},
    "models": [{"provider": "test", "model_name": "model", "cost_per_1m_input_tokens": 1.0, "cost_per_1m_output_tokens": 1.0, "error_rates": [0.1]}]
  })";
  EXPECT_THROW(RouterProfile::from_json_string(missing_metadata), std::exception);

  // Test malformed cluster centers
  std::string bad_centers = R"({
    "metadata": {"n_clusters": 1, "embedding_model": "test"},
    "cluster_centers": {"n_clusters": 1, "feature_dim": 1, "cluster_centers": "not_an_array"},
    "models": [{"provider": "test", "model_name": "model", "cost_per_1m_input_tokens": 1.0, "cost_per_1m_output_tokens": 1.0, "error_rates": [0.1]}]
  })";
  EXPECT_THROW(RouterProfile::from_json_string(bad_centers), std::exception);
}

TEST_F(ProfileTest, InvalidMsgpackParsing) {
  // Test invalid msgpack binary data
  std::string invalid_msgpack = "not msgpack data";
  EXPECT_THROW(RouterProfile::from_binary_string(invalid_msgpack), std::exception);

  // Test corrupted msgpack (valid msgpack but wrong structure)
  std::string corrupted_msgpack = "\x81\xa4test\x01";  // Simple map with one key-value, not our expected structure
  EXPECT_THROW(RouterProfile::from_binary_string(corrupted_msgpack), std::exception);
}

TEST_F(ProfileTest, FileOperationErrors) {
  // Test reading from non-existent JSON file
  EXPECT_THROW(RouterProfile::from_json("/nonexistent/file.json"), std::runtime_error);

  // Test reading from non-existent msgpack file
  EXPECT_THROW(RouterProfile::from_binary("/nonexistent/file.msgpack"), std::runtime_error);

  // Test writing to invalid path (though this might succeed on some systems)
  // This is harder to test reliably across platforms, so we'll skip it for now
}