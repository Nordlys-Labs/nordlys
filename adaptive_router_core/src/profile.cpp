#include "profile.hpp"

#include <cstdint>
#include <fstream>
#include <limits>
#include <msgpack.hpp>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>

using json = nlohmann::json;

// Define from_json for ClusteringConfig
void from_json(const json& j, ClusteringConfig& c) {
  c.max_iter = j.value("max_iter", 300);
  c.random_state = j.value("random_state", 42);
  c.n_init = j.value("n_init", 10);
  c.algorithm = j.value("algorithm", "lloyd");
  c.normalization_strategy = j.value("normalization_strategy", "l2");
}

// Define from_json for RoutingConfig
void from_json(const json& j, RoutingConfig& r) {
  r.lambda_min = j.value("lambda_min", 0.0f);
  r.lambda_max = j.value("lambda_max", 2.0f);
  r.default_cost_preference = j.value("default_cost_preference", 0.5f);
  r.max_alternatives = j.value("max_alternatives", 5);
}

// Define from_json for ModelFeatures
void from_json(const json& j, ModelFeatures& m) {
  j.at("provider").get_to(m.provider);
  j.at("model_name").get_to(m.model_name);
  m.model_id = m.provider + "/" + m.model_name;
  j.at("cost_per_1m_input_tokens").get_to(m.cost_per_1m_input_tokens);
  j.at("cost_per_1m_output_tokens").get_to(m.cost_per_1m_output_tokens);
  j.at("error_rates").get_to(m.error_rates);
}

// Define from_json for ProfileMetadata
void from_json(const json& j, ProfileMetadata& meta) {
  j.at("n_clusters").get_to(meta.n_clusters);
  j.at("embedding_model").get_to(meta.embedding_model);
  meta.silhouette_score = j.value("silhouette_score", 0.0f);

  if (j.contains("clustering")) {
    j.at("clustering").get_to(meta.clustering);
  }

  if (j.contains("routing")) {
    j.at("routing").get_to(meta.routing);
  }
}

RouterProfile RouterProfile::from_json(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open profile file: " + path);
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  return from_json_string(buffer.str());
}

RouterProfile RouterProfile::from_json_string(const std::string& json_str) {
  json j;
  try {
    j = json::parse(json_str);
  } catch (const json::parse_error& e) {
    throw std::invalid_argument(std::string("Failed to parse JSON: ") + e.what());
  }

  RouterProfile profile;

  try {
    // Parse cluster centers
    const auto& centers_json = j.at("cluster_centers");
    int n_clusters = centers_json.at("n_clusters").get<int>();
    int feature_dim = centers_json.at("feature_dim").get<int>();
    const auto& centers_data = centers_json.at("cluster_centers");

    // Validate n_clusters and feature_dim are positive
    if (n_clusters <= 0) {
      throw std::invalid_argument("n_clusters must be positive, got " + std::to_string(n_clusters));
    }
    if (feature_dim <= 0) {
      throw std::invalid_argument("feature_dim must be positive, got " + std::to_string(feature_dim));
    }

    // Check for overflow: promote to uint64_t before multiplication
    uint64_t total_elements = static_cast<uint64_t>(n_clusters) * static_cast<uint64_t>(feature_dim);
    uint64_t total_bytes = total_elements * sizeof(float);
    if (total_elements > static_cast<uint64_t>(std::numeric_limits<Eigen::Index>::max())) {
      throw std::invalid_argument("Cluster centers dimensions overflow: n_clusters="
                                  + std::to_string(n_clusters) + ", feature_dim="
                                  + std::to_string(feature_dim));
    }

    if (!centers_data.is_array()) {
      throw std::invalid_argument("cluster_centers.cluster_centers must be an array");
    }
    if (static_cast<int>(centers_data.size()) != n_clusters) {
      throw std::invalid_argument(
          "cluster_centers array size (" + std::to_string(centers_data.size())
          + ") does not match n_clusters (" + std::to_string(n_clusters) + ")");
    }

    profile.cluster_centers.resize(n_clusters, feature_dim);
    for (int i = 0; i < n_clusters; ++i) {
      if (!centers_data[i].is_array() || static_cast<int>(centers_data[i].size()) != feature_dim) {
        throw std::invalid_argument("Invalid cluster center dimensions at index "
                                    + std::to_string(i) + ": expected "
                                    + std::to_string(feature_dim) + " dimensions, got "
                                    + std::to_string(centers_data[i].size()));
      }
      for (int j_idx = 0; j_idx < feature_dim; ++j_idx) {
        profile.cluster_centers(i, j_idx) = centers_data[i][j_idx].get<float>();
      }
    }

    // Parse models (automatic via from_json)
    profile.models = j.at("models").get<std::vector<ModelFeatures>>();

    // Parse metadata (automatic via from_json)
    profile.metadata = j.at("metadata").get<ProfileMetadata>();

  } catch (const json::out_of_range& e) {
    throw std::invalid_argument(std::string("Missing required field in JSON: ") + e.what());
  } catch (const json::type_error& e) {
    throw std::invalid_argument(std::string("Invalid type in JSON: ") + e.what());
  }

  return profile;
}

RouterProfile RouterProfile::from_binary(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open binary profile file: " + path);
  }

  std::string buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

  msgpack::object_handle handle;
  try {
    handle = msgpack::unpack(buffer.data(), buffer.size());
  } catch (const std::exception& e) {
    throw std::invalid_argument(std::string("Failed to parse MessagePack: ") + e.what());
  }

  auto obj = handle.get();
  std::map<std::string, msgpack::object> map;

  try {
    map = obj.as<std::map<std::string, msgpack::object>>();
  } catch (const std::exception& e) {
    throw std::invalid_argument(std::string("Invalid MessagePack root structure: ") + e.what());
  }

  RouterProfile profile;

  try {
    // Parse cluster centers from raw bytes
    if (map.find("cluster_centers") == map.end()) {
      throw std::invalid_argument("Missing 'cluster_centers' in MessagePack data");
    }
    auto centers_map = map.at("cluster_centers").as<std::map<std::string, msgpack::object>>();

    if (centers_map.find("n_clusters") == centers_map.end()) {
      throw std::invalid_argument("Missing 'n_clusters' in cluster_centers");
    }
    if (centers_map.find("feature_dim") == centers_map.end()) {
      throw std::invalid_argument("Missing 'feature_dim' in cluster_centers");
    }
    if (centers_map.find("data") == centers_map.end()) {
      throw std::invalid_argument("Missing 'data' in cluster_centers");
    }

    int n_clusters = centers_map.at("n_clusters").as<int>();
    int feature_dim = centers_map.at("feature_dim").as<int>();
    std::string centers_bytes = centers_map.at("data").as<std::string>();

    // Validate n_clusters and feature_dim are positive
    if (n_clusters <= 0) {
      throw std::invalid_argument("n_clusters must be positive, got " + std::to_string(n_clusters));
    }
    if (feature_dim <= 0) {
      throw std::invalid_argument("feature_dim must be positive, got " + std::to_string(feature_dim));
    }

    // Check for overflow: promote to uint64_t before multiplication
    uint64_t total_elements = static_cast<uint64_t>(n_clusters) * static_cast<uint64_t>(feature_dim);
    uint64_t expected_size_u64 = total_elements * sizeof(float);
    if (total_elements > static_cast<uint64_t>(std::numeric_limits<Eigen::Index>::max())) {
      throw std::invalid_argument("Cluster centers dimensions overflow: n_clusters="
                                  + std::to_string(n_clusters) + ", feature_dim="
                                  + std::to_string(feature_dim));
    }

    // Safe to cast to size_t after validation
    size_t expected_size = static_cast<size_t>(expected_size_u64);
    if (centers_bytes.size() != expected_size) {
      throw std::invalid_argument("cluster_centers data size mismatch: expected "
                                  + std::to_string(expected_size) + " bytes, got "
                                  + std::to_string(centers_bytes.size()) + " bytes");
    }

    profile.cluster_centers.resize(n_clusters, feature_dim);
    std::memcpy(profile.cluster_centers.data(), centers_bytes.data(), expected_size);

    // Parse models
    if (map.find("models") == map.end()) {
      throw std::invalid_argument("Missing 'models' in MessagePack data");
    }
    auto models_arr = map.at("models").as<std::vector<msgpack::object>>();

    for (size_t idx = 0; idx < models_arr.size(); ++idx) {
      auto m = models_arr[idx].as<std::map<std::string, msgpack::object>>();
      ModelFeatures model;

      if (m.find("provider") == m.end()) {
        throw std::invalid_argument("Missing 'provider' in models[" + std::to_string(idx) + "]");
      }
      if (m.find("model_name") == m.end()) {
        throw std::invalid_argument("Missing 'model_name' in models[" + std::to_string(idx) + "]");
      }
      if (m.find("cost_per_1m_input_tokens") == m.end()) {
        throw std::invalid_argument("Missing 'cost_per_1m_input_tokens' in models["
                                    + std::to_string(idx) + "]");
      }
      if (m.find("cost_per_1m_output_tokens") == m.end()) {
        throw std::invalid_argument("Missing 'cost_per_1m_output_tokens' in models["
                                    + std::to_string(idx) + "]");
      }
      if (m.find("error_rates") == m.end()) {
        throw std::invalid_argument("Missing 'error_rates' in models[" + std::to_string(idx) + "]");
      }

      model.provider = m.at("provider").as<std::string>();
      model.model_name = m.at("model_name").as<std::string>();
      model.model_id = model.provider + "/" + model.model_name;
      model.cost_per_1m_input_tokens = m.at("cost_per_1m_input_tokens").as<float>();
      model.cost_per_1m_output_tokens = m.at("cost_per_1m_output_tokens").as<float>();
      model.error_rates = m.at("error_rates").as<std::vector<float>>();
      profile.models.push_back(std::move(model));
    }

    // Parse metadata
    if (map.find("metadata") == map.end()) {
      throw std::invalid_argument("Missing 'metadata' in MessagePack data");
    }
    auto meta = map.at("metadata").as<std::map<std::string, msgpack::object>>();

    if (meta.find("n_clusters") == meta.end()) {
      throw std::invalid_argument("Missing 'n_clusters' in metadata");
    }
    if (meta.find("embedding_model") == meta.end()) {
      throw std::invalid_argument("Missing 'embedding_model' in metadata");
    }

    profile.metadata.n_clusters = meta.at("n_clusters").as<int>();
    profile.metadata.embedding_model = meta.at("embedding_model").as<std::string>();

    if (meta.count("silhouette_score")) {
      profile.metadata.silhouette_score = meta.at("silhouette_score").as<float>();
    } else {
      profile.metadata.silhouette_score = 0.0f;
    }

  } catch (const std::exception& e) {
    throw std::invalid_argument(std::string("Error parsing MessagePack data: ") + e.what());
  }

  return profile;
}
