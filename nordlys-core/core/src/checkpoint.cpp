#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <format>
#include <fstream>
#include <limits>
#include <msgpack.hpp>
#include <nlohmann/json.hpp>
#include <nordlys_core/checkpoint.hpp>
#include <nordlys_core/tracy.hpp>
#include <ranges>
#include <sstream>
#include <stdexcept>

using json = nlohmann::json;

// ============================================================================
// JSON Serialization - TrainingMetrics
// ============================================================================

void to_json(json& j, const TrainingMetrics& m) {
  if (m.n_samples) j["n_samples"] = *m.n_samples;
  if (m.cluster_sizes) j["cluster_sizes"] = *m.cluster_sizes;
  if (m.silhouette_score) j["silhouette_score"] = *m.silhouette_score;
  if (m.inertia) j["inertia"] = *m.inertia;
}

void from_json(const json& j, TrainingMetrics& m) {
  if (j.contains("n_samples")) m.n_samples = j["n_samples"].get<int>();
  if (j.contains("cluster_sizes")) m.cluster_sizes = j["cluster_sizes"].get<std::vector<int>>();
  if (j.contains("silhouette_score")) m.silhouette_score = j["silhouette_score"].get<float>();
  if (j.contains("inertia")) m.inertia = j["inertia"].get<float>();
}

// ============================================================================
// JSON Serialization - EmbeddingConfig
// ============================================================================

void to_json(json& j, const EmbeddingConfig& c) {
  j = {{"model", c.model}, {"dtype", c.dtype}, {"trust_remote_code", c.trust_remote_code}};
}

void from_json(const json& j, EmbeddingConfig& c) {
  j.at("model").get_to(c.model);
  c.dtype = j.value("dtype", "float32");
  c.trust_remote_code = j.value("trust_remote_code", false);
}

// ============================================================================
// JSON Serialization - ClusteringConfig
// ============================================================================

void to_json(json& j, const ClusteringConfig& c) {
  j = {{"n_clusters", c.n_clusters}, {"random_state", c.random_state},
       {"max_iter", c.max_iter},     {"n_init", c.n_init},
       {"algorithm", c.algorithm},   {"normalization", c.normalization}};
}

void from_json(const json& j, ClusteringConfig& c) {
  j.at("n_clusters").get_to(c.n_clusters);
  c.random_state = j.value("random_state", 42);
  c.max_iter = j.value("max_iter", 300);
  c.n_init = j.value("n_init", 10);
  c.algorithm = j.value("algorithm", "lloyd");
  c.normalization = j.value("normalization", "l2");
}

// ============================================================================
// JSON Serialization - RoutingConfig
// ============================================================================

void to_json(json& j, const RoutingConfig& c) {
  j = {{"cost_bias_min", c.cost_bias_min},
       {"cost_bias_max", c.cost_bias_max},
       {"default_cost_bias", c.default_cost_bias},
       {"max_alternatives", c.max_alternatives}};
}

void from_json(const json& j, RoutingConfig& c) {
  c.cost_bias_min = j.value("cost_bias_min", 0.0f);
  c.cost_bias_max = j.value("cost_bias_max", 1.0f);
  c.default_cost_bias = j.value("default_cost_bias", 0.5f);
  c.max_alternatives = j.value("max_alternatives", 5);
}

// ============================================================================
// JSON Serialization - ModelFeatures
// ============================================================================

void to_json(json& j, const ModelFeatures& f) {
  j = {{"model_id", f.model_id},
       {"cost_per_1m_input_tokens", f.cost_per_1m_input_tokens},
       {"cost_per_1m_output_tokens", f.cost_per_1m_output_tokens},
       {"error_rates", f.error_rates}};
}

void from_json(const json& j, ModelFeatures& f) {
  j.at("model_id").get_to(f.model_id);
  j.at("cost_per_1m_input_tokens").get_to(f.cost_per_1m_input_tokens);
  j.at("cost_per_1m_output_tokens").get_to(f.cost_per_1m_output_tokens);
  j.at("error_rates").get_to(f.error_rates);
}

// ============================================================================
// JSON File I/O
// ============================================================================

NordlysCheckpoint NordlysCheckpoint::from_json(const std::string& path) {
  NORDLYS_ZONE;
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error(std::format("Failed to open checkpoint file: {}", path));
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  return from_json_string(buffer.str());
}

NordlysCheckpoint NordlysCheckpoint::from_json_string(const std::string& json_str) {
  NORDLYS_ZONE;
  json j = json::parse(json_str);

  NordlysCheckpoint checkpoint;

  // Version
  checkpoint.version = j.value("version", "2.0");

  // Configuration
  checkpoint.embedding = j.at("embedding").get<EmbeddingConfig>();
  checkpoint.clustering = j.at("clustering").get<ClusteringConfig>();
  checkpoint.routing = j.at("routing").get<RoutingConfig>();

  // Models
  checkpoint.models = j.at("models").get<std::vector<ModelFeatures>>();

  // Training metrics (optional)
  if (j.contains("metrics")) {
    checkpoint.metrics = j.at("metrics").get<TrainingMetrics>();
  }

  // Cluster centers
  const auto& centers_json = j.at("cluster_centers");
  auto n_clusters = static_cast<size_t>(checkpoint.clustering.n_clusters);
  size_t feature_dim = centers_json[0].size();

  if (checkpoint.embedding.dtype == "float64") {
    EmbeddingMatrixT<double> centers(static_cast<int>(n_clusters), static_cast<int>(feature_dim));
    for (size_t i = 0; i < n_clusters; ++i) {
      for (size_t col = 0; col < feature_dim; ++col) {
        centers(static_cast<int>(i), static_cast<int>(col)) = centers_json[i][col].get<double>();
      }
    }
    checkpoint.cluster_centers = std::move(centers);
  } else {
    EmbeddingMatrixT<float> centers(static_cast<int>(n_clusters), static_cast<int>(feature_dim));
    for (size_t i = 0; i < n_clusters; ++i) {
      for (size_t col = 0; col < feature_dim; ++col) {
        centers(static_cast<int>(i), static_cast<int>(col)) = centers_json[i][col].get<float>();
      }
    }
    checkpoint.cluster_centers = std::move(centers);
  }

  return checkpoint;
}

std::string NordlysCheckpoint::to_json_string() const {
  json j;

  j["version"] = version;

  // Configuration
  j["embedding"] = embedding;
  j["clustering"] = clustering;
  j["routing"] = routing;

  // Models
  j["models"] = models;

  // Training metrics
  j["metrics"] = metrics;

  // Cluster centers as 2D array
  std::visit(
      [&](const auto& centers) {
        json centers_array = json::array();
        for (Eigen::Index i = 0; i < centers.rows(); ++i) {
          json row = json::array();
          for (Eigen::Index col = 0; col < centers.cols(); ++col) {
            row.push_back(centers(i, col));
          }
          centers_array.push_back(row);
        }
        j["cluster_centers"] = centers_array;
      },
      cluster_centers);

  return j.dump(2);
}

void NordlysCheckpoint::to_json(const std::string& path) const {
  std::ofstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error(std::format("Failed to open checkpoint file for writing: {}", path));
  }
  file << to_json_string();
}

// ============================================================================
// MessagePack File I/O (with mmap)
// ============================================================================

NordlysCheckpoint NordlysCheckpoint::from_msgpack(const std::string& path) {
  NORDLYS_ZONE;

  // Open file
  int fd = open(path.c_str(), O_RDONLY);
  if (fd == -1) {
    throw std::runtime_error(std::format("Failed to open msgpack file: {}", path));
  }

  // Get file size
  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    close(fd);
    throw std::runtime_error(std::format("Failed to stat msgpack file: {}", path));
  }
  size_t file_size = static_cast<size_t>(sb.st_size);

  // Memory map the file
  void* mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (mapped == MAP_FAILED) {
    close(fd);
    throw std::runtime_error(std::format("Failed to mmap msgpack file: {}", path));
  }

  // Parse from mapped memory
  try {
    auto result = from_msgpack_string(std::string(static_cast<const char*>(mapped), file_size));
    munmap(mapped, file_size);
    close(fd);
    return result;
  } catch (...) {
    munmap(mapped, file_size);
    close(fd);
    throw;
  }
}

NordlysCheckpoint NordlysCheckpoint::from_msgpack_string(const std::string& data) {
  NORDLYS_ZONE;

  msgpack::object_handle handle = msgpack::unpack(data.data(), data.size());
  auto map = handle.get().as<std::map<std::string, msgpack::object>>();

  NordlysCheckpoint checkpoint;

  // Version
  checkpoint.version = map.contains("version") ? map.at("version").as<std::string>() : "2.0";

  // Embedding config
  auto emb = map.at("embedding").as<std::map<std::string, msgpack::object>>();
  checkpoint.embedding.model = emb.at("model").as<std::string>();
  checkpoint.embedding.dtype
      = emb.contains("dtype") ? emb.at("dtype").as<std::string>() : "float32";
  checkpoint.embedding.trust_remote_code
      = emb.contains("trust_remote_code") ? emb.at("trust_remote_code").as<bool>() : false;

  // Clustering config
  auto clust = map.at("clustering").as<std::map<std::string, msgpack::object>>();
  checkpoint.clustering.n_clusters = clust.at("n_clusters").as<int>();
  checkpoint.clustering.random_state
      = clust.contains("random_state") ? clust.at("random_state").as<int>() : 42;
  checkpoint.clustering.max_iter
      = clust.contains("max_iter") ? clust.at("max_iter").as<int>() : 300;
  checkpoint.clustering.n_init = clust.contains("n_init") ? clust.at("n_init").as<int>() : 10;
  checkpoint.clustering.algorithm
      = clust.contains("algorithm") ? clust.at("algorithm").as<std::string>() : "lloyd";
  checkpoint.clustering.normalization
      = clust.contains("normalization") ? clust.at("normalization").as<std::string>() : "l2";

  // Routing config
  auto rout = map.at("routing").as<std::map<std::string, msgpack::object>>();
  checkpoint.routing.cost_bias_min
      = rout.contains("cost_bias_min") ? rout.at("cost_bias_min").as<float>() : 0.0f;
  checkpoint.routing.cost_bias_max
      = rout.contains("cost_bias_max") ? rout.at("cost_bias_max").as<float>() : 1.0f;
  checkpoint.routing.default_cost_bias
      = rout.contains("default_cost_bias") ? rout.at("default_cost_bias").as<float>() : 0.5f;
  checkpoint.routing.max_alternatives
      = rout.contains("max_alternatives") ? rout.at("max_alternatives").as<int>() : 5;

  // Models
  auto models_arr = map.at("models").as<std::vector<msgpack::object>>();
  checkpoint.models.reserve(models_arr.size());
  for (const auto& model_obj : models_arr) {
    auto model_map = model_obj.as<std::map<std::string, msgpack::object>>();
    ModelFeatures model;
    model.model_id = model_map.at("model_id").as<std::string>();
    model.cost_per_1m_input_tokens = model_map.at("cost_per_1m_input_tokens").as<float>();
    model.cost_per_1m_output_tokens = model_map.at("cost_per_1m_output_tokens").as<float>();
    model.error_rates = model_map.at("error_rates").as<std::vector<float>>();
    checkpoint.models.push_back(std::move(model));
  }

  // Training metrics (optional)
  if (map.contains("metrics")) {
    auto met = map.at("metrics").as<std::map<std::string, msgpack::object>>();
    if (met.contains("n_samples")) checkpoint.metrics.n_samples = met.at("n_samples").as<int>();
    if (met.contains("cluster_sizes"))
      checkpoint.metrics.cluster_sizes = met.at("cluster_sizes").as<std::vector<int>>();
    if (met.contains("silhouette_score"))
      checkpoint.metrics.silhouette_score = met.at("silhouette_score").as<float>();
    if (met.contains("inertia")) checkpoint.metrics.inertia = met.at("inertia").as<float>();
  }

  // Cluster centers (binary blob)
  auto centers_map = map.at("cluster_centers").as<std::map<std::string, msgpack::object>>();
  int n_clusters = centers_map.at("rows").as<int>();
  int feature_dim = centers_map.at("cols").as<int>();
  std::string centers_bytes = centers_map.at("data").as<std::string>();

  uint64_t total_elements = static_cast<uint64_t>(n_clusters) * static_cast<uint64_t>(feature_dim);

  if (checkpoint.embedding.dtype == "float64") {
    size_t expected_size = total_elements * sizeof(double);
    if (centers_bytes.size() != expected_size) {
      throw std::invalid_argument(
          std::format("cluster_centers data size mismatch: expected {} bytes, got {}",
                      expected_size, centers_bytes.size()));
    }
    EmbeddingMatrixT<double> centers(n_clusters, feature_dim);
    std::memcpy(centers.data(), centers_bytes.data(), expected_size);
    checkpoint.cluster_centers = std::move(centers);
  } else {
    size_t expected_size = total_elements * sizeof(float);
    if (centers_bytes.size() != expected_size) {
      throw std::invalid_argument(
          std::format("cluster_centers data size mismatch: expected {} bytes, got {}",
                      expected_size, centers_bytes.size()));
    }
    EmbeddingMatrixT<float> centers(n_clusters, feature_dim);
    std::memcpy(centers.data(), centers_bytes.data(), expected_size);
    checkpoint.cluster_centers = std::move(centers);
  }

  return checkpoint;
}

std::string NordlysCheckpoint::to_msgpack_string() const {
  msgpack::sbuffer buffer;
  msgpack::packer<msgpack::sbuffer> pk(&buffer);

  // Top-level map with 7 keys
  pk.pack_map(7);

  // Version
  pk.pack("version");
  pk.pack(version);

  // Embedding config
  pk.pack("embedding");
  pk.pack_map(3);
  pk.pack("model");
  pk.pack(embedding.model);
  pk.pack("dtype");
  pk.pack(embedding.dtype);
  pk.pack("trust_remote_code");
  pk.pack(embedding.trust_remote_code);

  // Clustering config
  pk.pack("clustering");
  pk.pack_map(6);
  pk.pack("n_clusters");
  pk.pack(clustering.n_clusters);
  pk.pack("random_state");
  pk.pack(clustering.random_state);
  pk.pack("max_iter");
  pk.pack(clustering.max_iter);
  pk.pack("n_init");
  pk.pack(clustering.n_init);
  pk.pack("algorithm");
  pk.pack(clustering.algorithm);
  pk.pack("normalization");
  pk.pack(clustering.normalization);

  // Routing config
  pk.pack("routing");
  pk.pack_map(4);
  pk.pack("cost_bias_min");
  pk.pack(routing.cost_bias_min);
  pk.pack("cost_bias_max");
  pk.pack(routing.cost_bias_max);
  pk.pack("default_cost_bias");
  pk.pack(routing.default_cost_bias);
  pk.pack("max_alternatives");
  pk.pack(routing.max_alternatives);

  // Models
  pk.pack("models");
  pk.pack_array(static_cast<uint32_t>(models.size()));
  for (const auto& model : models) {
    pk.pack_map(4);
    pk.pack("model_id");
    pk.pack(model.model_id);
    pk.pack("cost_per_1m_input_tokens");
    pk.pack(model.cost_per_1m_input_tokens);
    pk.pack("cost_per_1m_output_tokens");
    pk.pack(model.cost_per_1m_output_tokens);
    pk.pack("error_rates");
    pk.pack(model.error_rates);
  }

  // Cluster centers (binary blob)
  pk.pack("cluster_centers");
  pk.pack_map(3);
  std::visit(
      [&](const auto& centers) {
        using Scalar = typename std::decay_t<decltype(centers)>::Scalar;
        pk.pack("rows");
        pk.pack(static_cast<int>(centers.rows()));
        pk.pack("cols");
        pk.pack(static_cast<int>(centers.cols()));
        pk.pack("data");
        size_t data_size = static_cast<size_t>(centers.rows()) * static_cast<size_t>(centers.cols())
                           * sizeof(Scalar);
        pk.pack_bin(static_cast<uint32_t>(data_size));
        pk.pack_bin_body(reinterpret_cast<const char*>(centers.data()),
                         static_cast<uint32_t>(data_size));
      },
      cluster_centers);

  // Training metrics (count non-null fields)
  pk.pack("metrics");
  int metrics_count = 0;
  if (metrics.n_samples) ++metrics_count;
  if (metrics.cluster_sizes) ++metrics_count;
  if (metrics.silhouette_score) ++metrics_count;
  if (metrics.inertia) ++metrics_count;

  pk.pack_map(static_cast<uint32_t>(metrics_count));
  if (metrics.n_samples) {
    pk.pack("n_samples");
    pk.pack(*metrics.n_samples);
  }
  if (metrics.cluster_sizes) {
    pk.pack("cluster_sizes");
    pk.pack(*metrics.cluster_sizes);
  }
  if (metrics.silhouette_score) {
    pk.pack("silhouette_score");
    pk.pack(*metrics.silhouette_score);
  }
  if (metrics.inertia) {
    pk.pack("inertia");
    pk.pack(*metrics.inertia);
  }

  return std::string(buffer.data(), buffer.size());
}

void NordlysCheckpoint::to_msgpack(const std::string& path) const {
  std::string binary_data = to_msgpack_string();
  std::ofstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error(std::format("Failed to open msgpack file for writing: {}", path));
  }
  file.write(binary_data.data(), static_cast<std::streamsize>(binary_data.size()));
}

// ============================================================================
// Validation
// ============================================================================

void NordlysCheckpoint::validate() const {
  // Validate clustering config
  if (clustering.n_clusters <= 0) {
    throw std::invalid_argument(
        std::format("n_clusters must be positive, got {}", clustering.n_clusters));
  }

  // Validate dtype
  if (embedding.dtype != "float32" && embedding.dtype != "float64") {
    throw std::invalid_argument(
        std::format("dtype must be 'float32' or 'float64', got '{}'", embedding.dtype));
  }

  // Validate cluster centers
  std::visit(
      [&](const auto& centers) {
        using Scalar = typename std::decay_t<decltype(centers)>::Scalar;
        bool is_double = std::is_same_v<Scalar, double>;

        if (is_double && embedding.dtype != "float64") {
          throw std::invalid_argument("Cluster centers are float64 but dtype is not 'float64'");
        }
        if (!is_double && embedding.dtype != "float32") {
          throw std::invalid_argument("Cluster centers are float32 but dtype is not 'float32'");
        }

        if (centers.rows() != clustering.n_clusters) {
          throw std::invalid_argument(
              std::format("Cluster centers rows ({}) does not match n_clusters ({})",
                          centers.rows(), clustering.n_clusters));
        }

        if (centers.cols() <= 0) {
          throw std::invalid_argument(
              std::format("feature_dim must be positive, got {}", centers.cols()));
        }
      },
      cluster_centers);

  // Validate models
  if (models.empty()) {
    throw std::invalid_argument("models array cannot be empty");
  }

  for (size_t i = 0; i < models.size(); ++i) {
    const auto& model = models[i];

    if (model.error_rates.size() != static_cast<size_t>(clustering.n_clusters)) {
      throw std::invalid_argument(
          std::format("Model {} error_rates size ({}) does not match n_clusters ({})", i,
                      model.error_rates.size(), clustering.n_clusters));
    }

    for (size_t j = 0; j < model.error_rates.size(); ++j) {
      float error_rate = model.error_rates[j];
      if (error_rate < 0.0f || error_rate > 1.0f) {
        throw std::invalid_argument(std::format(
            "Model {} error_rate[{}] ({}) must be in range [0.0, 1.0]", i, j, error_rate));
      }
    }

    if (model.cost_per_1m_input_tokens < 0.0f) {
      throw std::invalid_argument(
          std::format("Model {} cost_per_1m_input_tokens ({}) must be non-negative", i,
                      model.cost_per_1m_input_tokens));
    }

    if (model.cost_per_1m_output_tokens < 0.0f) {
      throw std::invalid_argument(
          std::format("Model {} cost_per_1m_output_tokens ({}) must be non-negative", i,
                      model.cost_per_1m_output_tokens));
    }
  }
}
