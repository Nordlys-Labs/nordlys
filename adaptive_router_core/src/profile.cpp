#include "profile.hpp"

#include <algorithm>
#include <cstdint>
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
    throw std::runtime_error(std::format("Failed to open profile file: {}", path));
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  return from_json_string(buffer.str());
}

RouterProfile RouterProfile::from_json_string(const std::string& json_str) {
  json j = json::parse(json_str);  // Let parse_error propagate naturally

  RouterProfile profile;

  // Parse cluster centers
  const auto& centers_json = j.at("cluster_centers");
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
  for (auto i : std::views::iota(std::size_t{0}, n_clusters_u)) {
    const auto& center = centers_data[i];

    if (!center.is_array() || center.size() != feature_dim_u) {
      throw std::invalid_argument(
        std::format("Invalid cluster center at index {}: expected {} dimensions, got {}", i, feature_dim, center.size())
      );
    }

    for (auto j : std::views::iota(std::size_t{0}, feature_dim_u)) {
      profile.cluster_centers(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = center[j].get<float>();
    }
  }

  // Parse models and metadata (automatic via from_json)
  profile.models = j.at("models").get<std::vector<ModelFeatures>>();
  profile.metadata = j.at("metadata").get<ProfileMetadata>();

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
  std::ranges::copy_n(
    reinterpret_cast<const float*>(centers_bytes.data()),
    static_cast<std::ptrdiff_t>(total_elements),
    profile.cluster_centers.data()
  );

  // Parse models
  auto models_arr = map.at("models").as<std::vector<msgpack::object>>();
  profile.models.reserve(models_arr.size());

  for (auto idx : std::views::iota(size_t{0}, models_arr.size())) {
    auto m = models_arr[idx].as<std::map<std::string, msgpack::object>>();
    ModelFeatures model;

    model.provider = m.at("provider").as<std::string>();
    model.model_name = m.at("model_name").as<std::string>();
    model.model_id = model.provider + "/" + model.model_name;
    model.cost_per_1m_input_tokens = m.at("cost_per_1m_input_tokens").as<float>();
    model.cost_per_1m_output_tokens = m.at("cost_per_1m_output_tokens").as<float>();
    model.error_rates = m.at("error_rates").as<std::vector<float>>();
    profile.models.push_back(std::move(model));
  }

  // Parse metadata
  auto meta = map.at("metadata").as<std::map<std::string, msgpack::object>>();
  profile.metadata.n_clusters = meta.at("n_clusters").as<int>();
  profile.metadata.embedding_model = meta.at("embedding_model").as<std::string>();
  profile.metadata.silhouette_score = meta.contains("silhouette_score") ? meta.at("silhouette_score").as<float>() : 0.0f;

  // Parse optional clustering config
  if (meta.contains("clustering")) {
    auto c = meta.at("clustering").as<std::map<std::string, msgpack::object>>();
    if (c.contains("max_iter")) profile.metadata.clustering.max_iter = c.at("max_iter").as<int>();
    if (c.contains("random_state")) profile.metadata.clustering.random_state = c.at("random_state").as<int>();
    if (c.contains("n_init")) profile.metadata.clustering.n_init = c.at("n_init").as<int>();
    if (c.contains("algorithm")) profile.metadata.clustering.algorithm = c.at("algorithm").as<std::string>();
    if (c.contains("normalization_strategy")) profile.metadata.clustering.normalization_strategy = c.at("normalization_strategy").as<std::string>();
  }

  // Parse optional routing config
  if (meta.contains("routing")) {
    auto r = meta.at("routing").as<std::map<std::string, msgpack::object>>();
    if (r.contains("lambda_min")) profile.metadata.routing.lambda_min = r.at("lambda_min").as<float>();
    if (r.contains("lambda_max")) profile.metadata.routing.lambda_max = r.at("lambda_max").as<float>();
    if (r.contains("default_cost_preference")) profile.metadata.routing.default_cost_preference = r.at("default_cost_preference").as<float>();
    if (r.contains("max_alternatives")) profile.metadata.routing.max_alternatives = r.at("max_alternatives").as<int>();
  }

  return profile;
}
