#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <nordlys_core/checkpoint.hpp>

namespace fs = std::filesystem;

class CheckpointIntegrationTest : public ::testing::Test {
protected:
  static fs::path get_test_data_dir() {
    return fs::path(__FILE__).parent_path().parent_path() / "fixtures";
  }

  static fs::path get_checkpoint_path(const std::string& name) {
    return get_test_data_dir() / name;
  }
};

TEST_F(CheckpointIntegrationTest, LoadFromJsonFile) {
  auto checkpoint_path = get_checkpoint_path("valid_checkpoint_f32.json");
  ASSERT_TRUE(fs::exists(checkpoint_path));

  auto checkpoint = NordlysCheckpoint::from_json(checkpoint_path.string());

  EXPECT_EQ(checkpoint.clustering.n_clusters, 3);
  EXPECT_EQ(checkpoint.embedding.model, "integration-test-model");
  EXPECT_EQ(checkpoint.models.size(), 3);
}

TEST_F(CheckpointIntegrationTest, JsonRoundTrip) {
  auto checkpoint_path = get_checkpoint_path("valid_checkpoint_f32.json");
  auto original = NordlysCheckpoint::from_json(checkpoint_path.string());

  auto json_str = original.to_json_string();

  auto loaded = NordlysCheckpoint::from_json_string(json_str);

  EXPECT_EQ(loaded.clustering.n_clusters, original.clustering.n_clusters);
  EXPECT_EQ(loaded.embedding.model, original.embedding.model);
  EXPECT_EQ(loaded.models.size(), original.models.size());
}

TEST_F(CheckpointIntegrationTest, MsgpackRoundTrip) {
  auto checkpoint_path = get_checkpoint_path("valid_checkpoint_f32.json");
  auto original = NordlysCheckpoint::from_json(checkpoint_path.string());

  auto msgpack_data = original.to_msgpack_string();

  auto loaded = NordlysCheckpoint::from_msgpack_string(msgpack_data);

  EXPECT_EQ(loaded.clustering.n_clusters, original.clustering.n_clusters);
  EXPECT_EQ(loaded.embedding.model, original.embedding.model);
  EXPECT_EQ(loaded.models.size(), original.models.size());
}

TEST_F(CheckpointIntegrationTest, JsonToMsgpackConversion) {
  auto json_path = get_checkpoint_path("valid_checkpoint_f32.json");
  auto from_json = NordlysCheckpoint::from_json(json_path.string());

  auto msgpack_data = from_json.to_msgpack_string();
  auto from_msgpack = NordlysCheckpoint::from_msgpack_string(msgpack_data);

  EXPECT_EQ(from_msgpack.clustering.n_clusters, from_json.clustering.n_clusters);
  EXPECT_EQ(from_msgpack.embedding.model, from_json.embedding.model);
  EXPECT_EQ(from_msgpack.models.size(), from_json.models.size());
}

TEST_F(CheckpointIntegrationTest, FileIOJsonRoundTrip) {
  auto original_path = get_checkpoint_path("valid_checkpoint_f32.json");
  auto original = NordlysCheckpoint::from_json(original_path.string());

  auto temp_path = fs::temp_directory_path() / "test_checkpoint_roundtrip.json";

  original.to_json(temp_path.string());
  ASSERT_TRUE(fs::exists(temp_path));

  auto loaded = NordlysCheckpoint::from_json(temp_path.string());

  EXPECT_EQ(loaded.clustering.n_clusters, original.clustering.n_clusters);

  fs::remove(temp_path);
}

TEST_F(CheckpointIntegrationTest, FileIOMsgpackRoundTrip) {
  auto original_path = get_checkpoint_path("valid_checkpoint_f32.json");
  auto original = NordlysCheckpoint::from_json(original_path.string());

  auto temp_path = fs::temp_directory_path() / "test_checkpoint_roundtrip.msgpack";

  original.to_msgpack(temp_path.string());
  ASSERT_TRUE(fs::exists(temp_path));

  auto loaded = NordlysCheckpoint::from_msgpack(temp_path.string());

  EXPECT_EQ(loaded.clustering.n_clusters, original.clustering.n_clusters);

  fs::remove(temp_path);
}

TEST_F(CheckpointIntegrationTest, ValidationPassesForValidCheckpoint) {
  auto checkpoint_path = get_checkpoint_path("valid_checkpoint_f32.json");
  auto checkpoint = NordlysCheckpoint::from_json(checkpoint_path.string());

  EXPECT_NO_THROW(checkpoint.validate());
}

TEST_F(CheckpointIntegrationTest, ValidationFailsForInvalidCheckpoint) {
  auto checkpoint_path = get_checkpoint_path("invalid_checkpoint.json");

  EXPECT_THROW(NordlysCheckpoint::from_json(checkpoint_path.string()), std::exception);
}

TEST_F(CheckpointIntegrationTest, InvalidJsonFileThrows) {
  EXPECT_THROW(NordlysCheckpoint::from_json("/nonexistent/file.json"), std::runtime_error);
}

TEST_F(CheckpointIntegrationTest, InvalidMsgpackFileThrows) {
  EXPECT_THROW(NordlysCheckpoint::from_msgpack("/nonexistent/file.msgpack"), std::runtime_error);
}

TEST_F(CheckpointIntegrationTest, CheckpointPropertiesAccessible) {
  auto checkpoint_path = get_checkpoint_path("valid_checkpoint_f32.json");
  auto checkpoint = NordlysCheckpoint::from_json(checkpoint_path.string());

  EXPECT_EQ(checkpoint.n_clusters(), 3);
  EXPECT_EQ(checkpoint.embedding_model(), "integration-test-model");
  EXPECT_FLOAT_EQ(checkpoint.silhouette_score(), 0.85f);
  EXPECT_FALSE(checkpoint.allow_trust_remote_code());

  EXPECT_EQ(checkpoint.clustering.algorithm, "lloyd");
  EXPECT_EQ(checkpoint.clustering.random_state, 42);
}
