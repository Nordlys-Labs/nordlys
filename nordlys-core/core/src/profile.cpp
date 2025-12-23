#include <nordlys_core/profile.hpp>

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

// Define to_json for ClusteringConfig
void to_json(json& json_obj, const ClusteringConfig& config) {
  json_obj = {
    {"max_iter", config.max_iter},
    {"random_state", config.random_state},
    {"n_init", config.n_init},
    {"algorithm", config.algorithm},
    {"normalization_strategy", config.normalization_strategy}
  };
}

// Define to_json for RoutingConfig
void to_json(json& json_obj, const RoutingConfig& config) {
  json_obj = {
    {"lambda_min", config.lambda_min},
    {"lambda_max", config.lambda_max},
    {"default_cost_preference", config.default_cost_preference},
    {"max_alternatives", config.max_alternatives}
  };
}

// Define to_json for ModelFeatures
void to_json(json& json_obj, const ModelFeatures& features) {
  json_obj = {
    {"provider", features.provider},
    {"model_name", features.model_name},
    {"cost_per_1m_input_tokens", features.cost_per_1m_input_tokens},
    {"cost_per_1m_output_tokens", features.cost_per_1m_output_tokens},
    {"error_rates", features.error_rates}
  };
}

// Define to_json for ProfileMetadata
void to_json(json& json_obj, const ProfileMetadata& meta) {
  json_obj = {
    {"n_clusters", meta.n_clusters},
    {"embedding_model", meta.embedding_model},
    {"dtype", meta.dtype},
    {"silhouette_score", meta.silhouette_score},
    {"allow_trust_remote_code", meta.allow_trust_remote_code},
    {"clustering", meta.clustering},
    {"routing", meta.routing}
  };
}

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
  meta.dtype = json_obj.value("dtype", "float32");  // Default float32 for backwards compat
  meta.silhouette_score = json_obj.value("silhouette_score", 0.0f);
  meta.allow_trust_remote_code = json_obj.value("allow_trust_remote_code", false);

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

  // Parse metadata first to get dtype
  profile.metadata = profile_json.at("metadata").get<ProfileMetadata>();

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

  // Parse cluster centers based on dtype
  if (profile.metadata.dtype == "float64") {
    EmbeddingMatrixT<double> centers(n_clusters, feature_dim);
    for (auto cluster_idx : std::views::iota(std::size_t{0}, n_clusters_u)) {
      const auto& center = centers_data[cluster_idx];
      if (!center.is_array() || center.size() != feature_dim_u) {
        throw std::invalid_argument(
          std::format("Invalid cluster center at index {}: expected {} dimensions, got {}", cluster_idx, feature_dim, center.size())
        );
      }
      for (auto col : std::views::iota(std::size_t{0}, feature_dim_u)) {
        centers(static_cast<Eigen::Index>(cluster_idx), static_cast<Eigen::Index>(col)) = center[col].get<double>();
      }
    }
    profile.cluster_centers = std::move(centers);
  } else {
    EmbeddingMatrixT<float> centers(n_clusters, feature_dim);
    for (auto cluster_idx : std::views::iota(std::size_t{0}, n_clusters_u)) {
      const auto& center = centers_data[cluster_idx];
      if (!center.is_array() || center.size() != feature_dim_u) {
        throw std::invalid_argument(
          std::format("Invalid cluster center at index {}: expected {} dimensions, got {}", cluster_idx, feature_dim, center.size())
        );
      }
      for (auto col : std::views::iota(std::size_t{0}, feature_dim_u)) {
        centers(static_cast<Eigen::Index>(cluster_idx), static_cast<Eigen::Index>(col)) = center[col].get<float>();
      }
    }
    profile.cluster_centers = std::move(centers);
  }

  // Parse models
  profile.models = profile_json.at("models").get<std::vector<ModelFeatures>>();

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

  // Parse metadata first to get dtype
  auto meta = map.at("metadata").as<std::map<std::string, msgpack::object>>();
  profile.metadata.n_clusters = meta.at("n_clusters").as<int>();
  profile.metadata.embedding_model = meta.at("embedding_model").as<std::string>();
  profile.metadata.dtype = meta.contains("dtype") ? meta.at("dtype").as<std::string>() : "float32";
  profile.metadata.silhouette_score = meta.contains("silhouette_score") ? meta.at("silhouette_score").as<float>() : 0.0f;
  profile.metadata.allow_trust_remote_code = meta.contains("allow_trust_remote_code") ? meta.at("allow_trust_remote_code").as<bool>() : false;

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

  // Parse cluster centers based on dtype
  if (profile.metadata.dtype == "float64") {
    size_t expected_size = total_elements * sizeof(double);
    if (centers_bytes.size() != expected_size) {
      throw std::invalid_argument(
        std::format("cluster_centers data size mismatch: expected {} bytes (float64), got {}", expected_size, centers_bytes.size())
      );
    }
    EmbeddingMatrixT<double> centers(n_clusters, feature_dim);
    std::memcpy(
      centers.data(),
      centers_bytes.data(),
      static_cast<size_t>(total_elements) * sizeof(double)
    );
    profile.cluster_centers = std::move(centers);
  } else {
    size_t expected_size = total_elements * sizeof(float);
    if (centers_bytes.size() != expected_size) {
      throw std::invalid_argument(
        std::format("cluster_centers data size mismatch: expected {} bytes (float32), got {}", expected_size, centers_bytes.size())
      );
    }
    EmbeddingMatrixT<float> centers(n_clusters, feature_dim);
    std::memcpy(
      centers.data(),
      centers_bytes.data(),
      static_cast<size_t>(total_elements) * sizeof(float)
    );
    profile.cluster_centers = std::move(centers);
  }

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

  return profile;
}

std::string RouterProfile::to_json_string() const {
  json profile_json;

  // Add metadata
  profile_json["metadata"] = metadata;

  // Add cluster centers
  json centers_json;
  centers_json["n_clusters"] = metadata.n_clusters;

  std::visit([&](const auto& centers) {
    using Scalar = typename std::decay_t<decltype(centers)>::Scalar;
    centers_json["feature_dim"] = centers.cols();
    centers_json["dtype"] = std::is_same_v<Scalar, double> ? "float64" : "float32";

    json centers_array = json::array();
    for (Eigen::Index i = 0; i < centers.rows(); ++i) {
      json center = json::array();
      for (Eigen::Index j = 0; j < centers.cols(); ++j) {
        center.push_back(centers(i, j));
      }
      centers_array.push_back(center);
    }
    centers_json["cluster_centers"] = centers_array;
  }, cluster_centers);

  profile_json["cluster_centers"] = centers_json;

  // Add models
  profile_json["models"] = models;

  return profile_json.dump(2);  // Pretty print with 2-space indent
}

void RouterProfile::to_json(const std::string& path) const {
  std::string json_str = to_json_string();
  std::ofstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error(std::format("Failed to open profile file for writing: {}", path));
  }
  file << json_str;
}

std::string RouterProfile::to_binary_string() const {
  msgpack::sbuffer buffer;
  msgpack::packer<msgpack::sbuffer> pk(&buffer);

  // Top-level map with 3 keys: metadata, cluster_centers, models
  pk.pack_map(3);

  // Pack metadata
  pk.pack("metadata");
  pk.pack_map(7);  // n_clusters, embedding_model, dtype, silhouette_score, allow_trust_remote_code, clustering, routing

  pk.pack("n_clusters");
  pk.pack(metadata.n_clusters);

  pk.pack("embedding_model");
  pk.pack(metadata.embedding_model);

  pk.pack("dtype");
  pk.pack(metadata.dtype);

  pk.pack("silhouette_score");
  pk.pack(metadata.silhouette_score);

  pk.pack("allow_trust_remote_code");
  pk.pack(metadata.allow_trust_remote_code);

  // Pack clustering config
  pk.pack("clustering");
  pk.pack_map(5);
  pk.pack("max_iter");
  pk.pack(metadata.clustering.max_iter);
  pk.pack("random_state");
  pk.pack(metadata.clustering.random_state);
  pk.pack("n_init");
  pk.pack(metadata.clustering.n_init);
  pk.pack("algorithm");
  pk.pack(metadata.clustering.algorithm);
  pk.pack("normalization_strategy");
  pk.pack(metadata.clustering.normalization_strategy);

  // Pack routing config
  pk.pack("routing");
  pk.pack_map(4);
  pk.pack("lambda_min");
  pk.pack(metadata.routing.lambda_min);
  pk.pack("lambda_max");
  pk.pack(metadata.routing.lambda_max);
  pk.pack("default_cost_preference");
  pk.pack(metadata.routing.default_cost_preference);
  pk.pack("max_alternatives");
  pk.pack(metadata.routing.max_alternatives);

  // Pack cluster centers
  pk.pack("cluster_centers");
  pk.pack_map(4);  // n_clusters, feature_dim, dtype, data

  pk.pack("n_clusters");
  pk.pack(metadata.n_clusters);

  std::visit([&](const auto& centers) {
    using Scalar = typename std::decay_t<decltype(centers)>::Scalar;
    pk.pack("feature_dim");
    pk.pack(static_cast<int>(centers.cols()));

    pk.pack("dtype");
    pk.pack(std::is_same_v<Scalar, double> ? "float64" : "float32");

    pk.pack("data");
    // Pack as binary data
    size_t data_size = static_cast<size_t>(centers.rows()) * static_cast<size_t>(centers.cols()) * sizeof(Scalar);
    if (data_size > std::numeric_limits<uint32_t>::max()) {
      throw std::overflow_error("Cluster centers data exceeds MessagePack bin32 limit");
    }
    pk.pack_bin(static_cast<uint32_t>(data_size));
    pk.pack_bin_body(reinterpret_cast<const char*>(centers.data()), static_cast<uint32_t>(data_size));
  }, cluster_centers);

  // Pack models
  pk.pack("models");
  pk.pack_array(static_cast<uint32_t>(models.size()));

  for (const auto& model : models) {
    pk.pack_map(5);
    pk.pack("provider");
    pk.pack(model.provider);
    pk.pack("model_name");
    pk.pack(model.model_name);
    pk.pack("cost_per_1m_input_tokens");
    pk.pack(model.cost_per_1m_input_tokens);
    pk.pack("cost_per_1m_output_tokens");
    pk.pack(model.cost_per_1m_output_tokens);
    pk.pack("error_rates");
    pk.pack(model.error_rates);
  }

  return std::string(buffer.data(), buffer.size());
}

void RouterProfile::to_binary(const std::string& path) const {
  std::string binary_data = to_binary_string();
  std::ofstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error(std::format("Failed to open binary profile file for writing: {}", path));
  }
  file.write(binary_data.data(), static_cast<std::streamsize>(binary_data.size()));
}

RouterProfile RouterProfile::from_binary_string(const std::string& data) {
  msgpack::object_handle handle = msgpack::unpack(data.data(), data.size());
  auto map = handle.get().as<std::map<std::string, msgpack::object>>();

  RouterProfile profile;

  // Parse metadata first to get dtype
  auto meta = map.at("metadata").as<std::map<std::string, msgpack::object>>();
  profile.metadata.n_clusters = meta.at("n_clusters").as<int>();
  profile.metadata.embedding_model = meta.at("embedding_model").as<std::string>();
  profile.metadata.dtype = meta.contains("dtype") ? meta.at("dtype").as<std::string>() : "float32";
  profile.metadata.silhouette_score = meta.contains("silhouette_score") ? meta.at("silhouette_score").as<float>() : 0.0f;
  profile.metadata.allow_trust_remote_code = meta.contains("allow_trust_remote_code") ? meta.at("allow_trust_remote_code").as<bool>() : false;

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

  // Parse cluster centers based on dtype
  if (profile.metadata.dtype == "float64") {
    size_t expected_size = total_elements * sizeof(double);
    if (centers_bytes.size() != expected_size) {
      throw std::invalid_argument(
        std::format("cluster_centers data size mismatch: expected {} bytes (float64), got {}", expected_size, centers_bytes.size())
      );
    }
    EmbeddingMatrixT<double> centers(n_clusters, feature_dim);
    std::memcpy(
      centers.data(),
      centers_bytes.data(),
      static_cast<size_t>(total_elements) * sizeof(double)
    );
    profile.cluster_centers = std::move(centers);
  } else {
    size_t expected_size = total_elements * sizeof(float);
    if (centers_bytes.size() != expected_size) {
      throw std::invalid_argument(
        std::format("cluster_centers data size mismatch: expected {} bytes (float32), got {}", expected_size, centers_bytes.size())
      );
    }
    EmbeddingMatrixT<float> centers(n_clusters, feature_dim);
    std::memcpy(
      centers.data(),
      centers_bytes.data(),
      static_cast<size_t>(total_elements) * sizeof(float)
    );
    profile.cluster_centers = std::move(centers);
  }

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

  return profile;
}

void RouterProfile::validate() const {
  // Validate metadata
  if (metadata.n_clusters <= 0) {
    throw std::invalid_argument(std::format("n_clusters must be positive, got {}", metadata.n_clusters));
  }

  if (metadata.dtype != "float32" && metadata.dtype != "float64") {
    throw std::invalid_argument(std::format("dtype must be 'float32' or 'float64', got '{}'", metadata.dtype));
  }

  // Validate cluster centers
  std::visit([&](const auto& centers) {
    using Scalar = typename std::decay_t<decltype(centers)>::Scalar;
    bool is_double = std::is_same_v<Scalar, double>;

    if (is_double && metadata.dtype != "float64") {
      throw std::invalid_argument("Cluster centers are float64 but metadata.dtype is not 'float64'");
    }
    if (!is_double && metadata.dtype != "float32") {
      throw std::invalid_argument("Cluster centers are float32 but metadata.dtype is not 'float32'");
    }

    if (centers.rows() != metadata.n_clusters) {
      throw std::invalid_argument(std::format(
        "Cluster centers rows ({}) does not match n_clusters ({})", centers.rows(), metadata.n_clusters
      ));
    }

    if (centers.cols() <= 0) {
      throw std::invalid_argument(std::format("feature_dim must be positive, got {}", centers.cols()));
    }
  }, cluster_centers);

  // Validate models
  if (models.empty()) {
    throw std::invalid_argument("models array cannot be empty");
  }

  for (size_t i = 0; i < models.size(); ++i) {
    const auto& model = models[i];

    if (model.error_rates.size() != static_cast<size_t>(metadata.n_clusters)) {
      throw std::invalid_argument(std::format(
        "Model {} error_rates size ({}) does not match n_clusters ({})",
        i, model.error_rates.size(), metadata.n_clusters
      ));
    }

    for (size_t j = 0; j < model.error_rates.size(); ++j) {
      float error_rate = model.error_rates[j];
      if (error_rate < 0.0f || error_rate > 1.0f) {
        throw std::invalid_argument(std::format(
          "Model {} error_rate[{}] ({}) must be in range [0.0, 1.0]",
          i, j, error_rate
        ));
      }
    }

    if (model.cost_per_1m_input_tokens < 0.0f) {
      throw std::invalid_argument(std::format(
        "Model {} cost_per_1m_input_tokens ({}) must be non-negative",
        i, model.cost_per_1m_input_tokens
      ));
    }

    if (model.cost_per_1m_output_tokens < 0.0f) {
      throw std::invalid_argument(std::format(
        "Model {} cost_per_1m_output_tokens ({}) must be non-negative",
        i, model.cost_per_1m_output_tokens
      ));
    }
  }
}
