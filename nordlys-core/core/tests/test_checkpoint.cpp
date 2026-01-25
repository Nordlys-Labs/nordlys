#include <gtest/gtest.h>

#ifdef _WIN32
#  include <process.h>
#  define GETPID() _getpid()
#else
#  include <unistd.h>
#  define GETPID() getpid()
#endif

#include <cstdio>
#include <filesystem>
#include <nordlys_core/checkpoint.hpp>
#include <random>

#pragma GCC diagnostic ignored "-Wunused-result"

static const char* kTestCheckpointJson = R"({
  "version": "2.0",
  "cluster_centers": [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0]
  ],
  "models": [
    {
      "model_id": "openai/gpt-4",
      "cost_per_1m_input_tokens": 30.0,
      "cost_per_1m_output_tokens": 60.0,
      "error_rates": [0.1, 0.05, 0.15]
    },
    {
      "model_id": "anthropic/claude",
      "cost_per_1m_input_tokens": 15.0,
      "cost_per_1m_output_tokens": 45.0,
      "error_rates": [0.08, 0.12, 0.06]
    },
    {
      "model_id": "google/gemini-pro",
      "cost_per_1m_input_tokens": 0.5,
      "cost_per_1m_output_tokens": 1.5,
      "error_rates": [0.2, 0.18, 0.25]
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
  "metrics": {
    "silhouette_score": 0.85
  }
})";

class ProfileTest : public ::testing::Test {
protected:
  NordlysCheckpoint test_profile = NordlysCheckpoint::from_json_string(kTestCheckpointJson);
};

TEST_F(ProfileTest, RoundTripJson) {
  std::string json_str = test_profile.to_json_string();
  NordlysCheckpoint loaded = NordlysCheckpoint::from_json_string(json_str);

  EXPECT_EQ(loaded.clustering.n_clusters, test_profile.clustering.n_clusters);
  EXPECT_EQ(loaded.embedding.model, test_profile.embedding.model);
  EXPECT_EQ(loaded.metrics.silhouette_score, test_profile.metrics.silhouette_score);

  EXPECT_EQ(loaded.clustering.max_iter, test_profile.clustering.max_iter);
  EXPECT_EQ(loaded.clustering.algorithm, test_profile.clustering.algorithm);

  EXPECT_EQ(test_profile.cluster_centers.rows(), loaded.cluster_centers.rows());
  EXPECT_EQ(test_profile.cluster_centers.cols(), loaded.cluster_centers.cols());
  for (size_t i = 0; i < test_profile.cluster_centers.rows(); ++i) {
    for (size_t j = 0; j < test_profile.cluster_centers.cols(); ++j) {
      EXPECT_FLOAT_EQ(test_profile.cluster_centers(i, j), loaded.cluster_centers(i, j));
    }
  }

  EXPECT_EQ(loaded.models.size(), test_profile.models.size());
  for (size_t i = 0; i < loaded.models.size(); ++i) {
    EXPECT_EQ(loaded.models[i].model_id, test_profile.models[i].model_id);
    EXPECT_FLOAT_EQ(loaded.models[i].cost_per_1m_input_tokens,
                    test_profile.models[i].cost_per_1m_input_tokens);
    EXPECT_FLOAT_EQ(loaded.models[i].cost_per_1m_output_tokens,
                    test_profile.models[i].cost_per_1m_output_tokens);
    EXPECT_EQ(loaded.models[i].error_rates.size(), test_profile.models[i].error_rates.size());
    for (size_t j = 0; j < loaded.models[i].error_rates.size(); ++j) {
      EXPECT_FLOAT_EQ(loaded.models[i].error_rates[j], test_profile.models[i].error_rates[j]);
    }
  }
}

TEST_F(ProfileTest, RoundTripMsgpack) {
  std::string msgpack_data = test_profile.to_msgpack_string();
  NordlysCheckpoint loaded = NordlysCheckpoint::from_msgpack_string(msgpack_data);

  EXPECT_EQ(loaded.clustering.n_clusters, test_profile.clustering.n_clusters);
  EXPECT_EQ(loaded.embedding.model, test_profile.embedding.model);
  EXPECT_EQ(loaded.metrics.silhouette_score, test_profile.metrics.silhouette_score);

  EXPECT_EQ(test_profile.cluster_centers.rows(), loaded.cluster_centers.rows());
  EXPECT_EQ(test_profile.cluster_centers.cols(), loaded.cluster_centers.cols());
  for (size_t i = 0; i < test_profile.cluster_centers.rows(); ++i) {
    for (size_t j = 0; j < test_profile.cluster_centers.cols(); ++j) {
      EXPECT_FLOAT_EQ(test_profile.cluster_centers(i, j), loaded.cluster_centers(i, j));
    }
  }

  EXPECT_EQ(loaded.models.size(), test_profile.models.size());
  for (size_t i = 0; i < loaded.models.size(); ++i) {
    EXPECT_EQ(loaded.models[i].model_id, test_profile.models[i].model_id);
    EXPECT_EQ(loaded.models[i].error_rates.size(), test_profile.models[i].error_rates.size());
  }
}

TEST_F(ProfileTest, FileOperations) {
  auto temp_dir = std::filesystem::temp_directory_path();
  auto pid_suffix = std::to_string(GETPID());
  std::string json_file = (temp_dir / ("test_profile_" + pid_suffix + ".json")).string();
  test_profile.to_json(json_file);

  NordlysCheckpoint loaded_json = NordlysCheckpoint::from_json(json_file);

  EXPECT_EQ(loaded_json.clustering.n_clusters, test_profile.clustering.n_clusters);
  EXPECT_EQ(loaded_json.models.size(), test_profile.models.size());

  std::string msgpack_file = (temp_dir / ("test_profile_" + pid_suffix + ".msgpack")).string();
  test_profile.to_msgpack(msgpack_file);

  NordlysCheckpoint loaded_msgpack = NordlysCheckpoint::from_msgpack(msgpack_file);

  EXPECT_EQ(loaded_msgpack.clustering.n_clusters, test_profile.clustering.n_clusters);
  EXPECT_EQ(loaded_msgpack.models.size(), test_profile.models.size());

  std::remove(json_file.c_str());
  std::remove(msgpack_file.c_str());
}

TEST_F(ProfileTest, Validation) {
  EXPECT_NO_THROW(test_profile.validate());

  NordlysCheckpoint invalid_profile = test_profile;
  invalid_profile.clustering.n_clusters = -1;
  EXPECT_THROW(invalid_profile.validate(), std::invalid_argument);

  invalid_profile = test_profile;
  invalid_profile.models[0].error_rates.resize(2);
  EXPECT_THROW(invalid_profile.validate(), std::invalid_argument);

  invalid_profile = test_profile;
  invalid_profile.models[0].error_rates[0] = 1.5f;
  EXPECT_THROW(invalid_profile.validate(), std::invalid_argument);

  invalid_profile = test_profile;
  invalid_profile.models[0].cost_per_1m_input_tokens = -1.0f;
  EXPECT_THROW(invalid_profile.validate(), std::invalid_argument);
}

TEST_F(ProfileTest, InvalidJsonParsing) {
  std::string invalid_json = "{ invalid json }";
  EXPECT_THROW(NordlysCheckpoint::from_json_string(invalid_json), std::exception);

  std::string missing_embedding = R"({
    "version": "2.0",
    "cluster_centers": [[1.0]],
    "models": [{"model_id": "test/model", "cost_per_1m_input_tokens": 1.0, "cost_per_1m_output_tokens": 1.0, "error_rates": [0.1]}],
    "clustering": {"n_clusters": 1}
  })";
  EXPECT_THROW(NordlysCheckpoint::from_json_string(missing_embedding), std::exception);

  std::string bad_centers = R"({
    "version": "2.0",
    "cluster_centers": "not_an_array",
    "models": [{"model_id": "test/model", "cost_per_1m_input_tokens": 1.0, "cost_per_1m_output_tokens": 1.0, "error_rates": [0.1]}],
    "embedding": {"model": "test"},
    "clustering": {"n_clusters": 1}
  })";
  EXPECT_THROW(NordlysCheckpoint::from_json_string(bad_centers), std::exception);
}

TEST_F(ProfileTest, InvalidMsgpackParsing) {
  std::string invalid_msgpack = "not msgpack data";
  EXPECT_THROW(NordlysCheckpoint::from_msgpack_string(invalid_msgpack), std::exception);

  std::string corrupted_msgpack = "\x81\xa4test\x01";
  EXPECT_THROW(NordlysCheckpoint::from_msgpack_string(corrupted_msgpack), std::exception);
}

TEST_F(ProfileTest, FileOperationErrors) {
  EXPECT_THROW(NordlysCheckpoint::from_json("/nonexistent/file.json"), std::runtime_error);
  EXPECT_THROW(NordlysCheckpoint::from_msgpack("/nonexistent/file.msgpack"), std::runtime_error);
}

TEST_F(ProfileTest, ConvenienceAccessors) {
  EXPECT_EQ(test_profile.embedding_model(), "test-model");
  EXPECT_EQ(test_profile.random_state(), 42);
  EXPECT_EQ(test_profile.allow_trust_remote_code(), false);
  EXPECT_EQ(test_profile.n_clusters(), 3);
  EXPECT_EQ(test_profile.feature_dim(), 4);
  EXPECT_FLOAT_EQ(test_profile.silhouette_score(), 0.85f);
}

TEST_F(ProfileTest, CopyConstructor) {
  NordlysCheckpoint copy(test_profile);
  EXPECT_EQ(copy.clustering.n_clusters, test_profile.clustering.n_clusters);
  EXPECT_EQ(copy.embedding.model, test_profile.embedding.model);
  EXPECT_EQ(copy.models.size(), test_profile.models.size());
}

TEST_F(ProfileTest, CopyAssignment) {
  NordlysCheckpoint copy = NordlysCheckpoint::from_json_string(kTestCheckpointJson);
  copy = test_profile;
  EXPECT_EQ(copy.clustering.n_clusters, test_profile.clustering.n_clusters);
}

TEST_F(ProfileTest, MoveConstructor) {
  NordlysCheckpoint original = test_profile;
  NordlysCheckpoint moved(std::move(original));
  EXPECT_EQ(moved.clustering.n_clusters, 3);
  EXPECT_EQ(moved.models.size(), 3);
}

TEST_F(ProfileTest, MoveAssignment) {
  NordlysCheckpoint original = test_profile;
  NordlysCheckpoint moved = NordlysCheckpoint::from_json_string(kTestCheckpointJson);
  moved = std::move(original);
  EXPECT_EQ(moved.clustering.n_clusters, 3);
  EXPECT_EQ(moved.models.size(), 3);
}

TEST_F(ProfileTest, LargeNumberOfModels) {
  std::stringstream ss;
  ss << R"({"version": "2.0", "cluster_centers": [[1.0, 0.0]], "models": [)";
  for (int i = 0; i < 100; ++i) {
    if (i > 0) ss << ",";
    ss << R"({"model_id": "provider/model)" << i << R"(", "cost_per_1m_input_tokens": )" << i
       << R"(, "cost_per_1m_output_tokens": )" << (i * 2) << R"(, "error_rates": [0.01]})";
  }
  ss << R"(], "embedding": {"model": "test", "trust_remote_code": false}, )"
     << R"("clustering": {"n_clusters": 1, "random_state": 42, "max_iter": 300, "n_init": 10, "algorithm": "lloyd", "normalization": "l2"}, )"
     << R"("metrics": {"silhouette_score": 0.5}})";

  NordlysCheckpoint profile = NordlysCheckpoint::from_json_string(ss.str());
  EXPECT_EQ(profile.models.size(), 100);
  EXPECT_EQ(profile.models[99].model_id, "provider/model99");
}

TEST_F(ProfileTest, ValidationErrorRateOutOfRange) {
  NordlysCheckpoint invalid_profile = test_profile;
  invalid_profile.models[0].error_rates[0] = -0.1f;
  EXPECT_THROW(invalid_profile.validate(), std::invalid_argument);
}

TEST_F(ProfileTest, ValidationZeroModels) {
  NordlysCheckpoint invalid_profile = test_profile;
  invalid_profile.models.clear();
  EXPECT_THROW(invalid_profile.validate(), std::invalid_argument);
}
