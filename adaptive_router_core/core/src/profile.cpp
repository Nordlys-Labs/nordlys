#include <adaptive_core/profile.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <format>
#include <fstream>
#include <limits>
#include <msgpack.hpp>
#include <nlohmann/json.hpp>
#include <ranges>
#include <sstream>
#include <stdexcept>

using json = nlohmann::json;

// Define from_json for ClusteringConfig
void from_json(const json& json_obj, ClusteringConfig& config) {
  config.max_iter = json_obj.value("max_iter", 300);
  config.random_state = json_obj.value("random_state", 42);
  config.n_init = json_obj.value("n_init", 10);
  config.algorithm = json_obj.value("algorithm", "lloyd");
  config.normalization_strategy = json_obj.value("normalization_strategy", "l2");
}

// Define from_json for RoutingConfig
void from_json(const json& json_obj, RoutingConfig& config) {
  config.lambda_min = json_obj.value("lambda_min", 0.0f);
  config.lambda_max = json_obj.value("lambda_max", 2.0f);
  config.default_cost_preference = json_obj.value("default_cost_preference", 0.5f);
  config.max_alternatives = json_obj.value("max_alternatives", 5);
}

// Define from_json for ModelFeatures
void from_json(const json& json_obj, ModelFeatures& features) {
  json_obj.at("provider").get_to(features.provider);
  json_obj.at("model_name").get_to(features.model_name);
  features.model_id = features.provider + "/" + features.model_name;
  json_obj.at("cost_per_1m_input_tokens").get_to(features.cost_per_1m_input_tokens);
  json_obj.at("cost_per_1m_output_tokens").get_to(features.cost_per_1m_output_tokens);
  json_obj.at("error_rates").get_to(features.error_rates);
}

// Define from_json for ProfileMetadata
void from_json(const json& json_obj, ProfileMetadata& meta) {
  json_obj.at("n_clusters").get_to(meta.n_clusters);
  json_obj.at("embedding_model").get_to(meta.embedding_model);
  meta.silhouette_score = json_obj.value("silhouette_score", 0.0f);

  if (json_obj.contains("clustering")) {
    json_obj.at("clustering").get_to(meta.clustering);
  }

  if (json_obj.contains("routing")) {
    json_obj.at("routing").get_to(meta.routing);
  }
}

RouterProfile RouterProfile::from_json(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error(std::format("Failed to open profile file: {}", path));
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  return from_json_string(buffer.str());
}

RouterProfile RouterProfile::from_json_string(const std::string& json_str) {
  json profile_json = json::parse(json_str);  // Let parse_error propagate naturally

  RouterProfile profile;

  // Parse cluster centers
  const auto& centers_json = profile_json.at("cluster_centers");
  int n_clusters = centers_json.at("n_clusters").get<int>();
  int feature_dim = centers_json.at("feature_dim").get<int>();
  const auto& centers_data = centers_json.at("cluster_centers");

  // Validate dimensions
  if (n_clusters <= 0) {
    throw std::invalid_argument(std::format("n_clusters must be positive, got {}", n_clusters));
  }
  if (feature_dim <= 0) {
    throw std::invalid_argument(std::format("feature_dim must be positive, got {}", feature_dim));
  }

  // Check for overflow
  uint64_t total_elements = static_cast<uint64_t>(n_clusters) * static_cast<uint64_t>(feature_dim);
  if (total_elements > static_cast<uint64_t>(std::numeric_limits<Eigen::Index>::max())) {
    throw std::invalid_argument(
      std::format("Cluster centers dimensions overflow: n_clusters={}, feature_dim={}", n_clusters, feature_dim)
    );
  }

  if (!centers_data.is_array() || static_cast<int>(centers_data.size()) != n_clusters) {
    throw std::invalid_argument(
      std::format("cluster_centers array size ({}) does not match n_clusters ({})", centers_data.size(), n_clusters)
    );
  }

  auto n_clusters_u = static_cast<std::size_t>(n_clusters);
  auto feature_dim_u = static_cast<std::size_t>(feature_dim);

  profile.cluster_centers.resize(n_clusters, feature_dim);

  // Parse cluster centers using ranges
  for (auto cluster_idx : std::views::iota(std::size_t{0}, n_clusters_u)) {
    const auto& center = centers_data[cluster_idx];

    if (!center.is_array() || center.size() != feature_dim_u) {
      throw std::invalid_argument(
        std::format("Invalid cluster center at index {}: expected {} dimensions, got {}", cluster_idx, feature_dim, center.size())
      );
    }

    for (auto col : std::views::iota(std::size_t{0}, feature_dim_u)) {
      profile.cluster_centers(static_cast<Eigen::Index>(cluster_idx), static_cast<Eigen::Index>(col)) = center[col].get<float>();
    }
  }

  // Parse models and metadata (automatic via from_json)
  profile.models = profile_json.at("models").get<std::vector<ModelFeatures>>();
  profile.metadata = profile_json.at("metadata").get<ProfileMetadata>();

  return profile;
}

RouterProfile RouterProfile::from_binary(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error(std::format("Failed to open binary profile file: {}", path));
  }

  std::string buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  msgpack::object_handle handle = msgpack::unpack(buffer.data(), buffer.size());
  auto map = handle.get().as<std::map<std::string, msgpack::object>>();

  RouterProfile profile;

  // Parse cluster centers
  auto centers_map = map.at("cluster_centers").as<std::map<std::string, msgpack::object>>();
  int n_clusters = centers_map.at("n_clusters").as<int>();
  int feature_dim = centers_map.at("feature_dim").as<int>();
  std::string centers_bytes = centers_map.at("data").as<std::string>();

  // Validate dimensions
  if (n_clusters <= 0) {
    throw std::invalid_argument(std::format("n_clusters must be positive, got {}", n_clusters));
  }
  if (feature_dim <= 0) {
    throw std::invalid_argument(std::format("feature_dim must be positive, got {}", feature_dim));
  }

  // Check for overflow
  uint64_t total_elements = static_cast<uint64_t>(n_clusters) * static_cast<uint64_t>(feature_dim);
  if (total_elements > static_cast<uint64_t>(std::numeric_limits<Eigen::Index>::max())) {
    throw std::invalid_argument(
      std::format("Cluster centers dimensions overflow: n_clusters={}, feature_dim={}", n_clusters, feature_dim)
    );
  }

  size_t expected_size = total_elements * sizeof(float);
  if (centers_bytes.size() != expected_size) {
    throw std::invalid_argument(
      std::format("cluster_centers data size mismatch: expected {} bytes, got {}", expected_size, centers_bytes.size())
    );
  }

  profile.cluster_centers.resize(n_clusters, feature_dim);
  std::memcpy(
    profile.cluster_centers.data(),
    centers_bytes.data(),
    static_cast<size_t>(total_elements) * sizeof(float)
  );

  // Parse models
  auto models_arr = map.at("models").as<std::vector<msgpack::object>>();
  profile.models.reserve(models_arr.size());

  for (auto idx : std::views::iota(size_t{0}, models_arr.size())) {
    auto model_map = models_arr[idx].as<std::map<std::string, msgpack::object>>();
    ModelFeatures model;

    model.provider = model_map.at("provider").as<std::string>();
    model.model_name = model_map.at("model_name").as<std::string>();
    model.model_id = model.provider + "/" + model.model_name;
    model.cost_per_1m_input_tokens = model_map.at("cost_per_1m_input_tokens").as<float>();
    model.cost_per_1m_output_tokens = model_map.at("cost_per_1m_output_tokens").as<float>();
    model.error_rates = model_map.at("error_rates").as<std::vector<float>>();
    profile.models.push_back(std::move(model));
  }

  // Parse metadata
  auto meta = map.at("metadata").as<std::map<std::string, msgpack::object>>();
  profile.metadata.n_clusters = meta.at("n_clusters").as<int>();
  profile.metadata.embedding_model = meta.at("embedding_model").as<std::string>();
  profile.metadata.silhouette_score = meta.contains("silhouette_score") ? meta.at("silhouette_score").as<float>() : 0.0f;

  // Parse optional clustering config
  if (meta.contains("clustering")) {
    auto clustering_map = meta.at("clustering").as<std::map<std::string, msgpack::object>>();
    if (clustering_map.contains("max_iter")) profile.metadata.clustering.max_iter = clustering_map.at("max_iter").as<int>();
    if (clustering_map.contains("random_state")) profile.metadata.clustering.random_state = clustering_map.at("random_state").as<int>();
    if (clustering_map.contains("n_init")) profile.metadata.clustering.n_init = clustering_map.at("n_init").as<int>();
    if (clustering_map.contains("algorithm")) profile.metadata.clustering.algorithm = clustering_map.at("algorithm").as<std::string>();
    if (clustering_map.contains("normalization_strategy")) profile.metadata.clustering.normalization_strategy = clustering_map.at("normalization_strategy").as<std::string>();
  }

  // Parse optional routing config
  if (meta.contains("routing")) {
    auto routing_map = meta.at("routing").as<std::map<std::string, msgpack::object>>();
    if (routing_map.contains("lambda_min")) profile.metadata.routing.lambda_min = routing_map.at("lambda_min").as<float>();
    if (routing_map.contains("lambda_max")) profile.metadata.routing.lambda_max = routing_map.at("lambda_max").as<float>();
    if (routing_map.contains("default_cost_preference")) profile.metadata.routing.default_cost_preference = routing_map.at("default_cost_preference").as<float>();
    if (routing_map.contains("max_alternatives")) profile.metadata.routing.max_alternatives = routing_map.at("max_alternatives").as<int>();
  }

  return profile;
}
