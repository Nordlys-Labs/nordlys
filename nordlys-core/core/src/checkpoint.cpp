#ifdef _WIN32
#include <io.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

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
  if (j.contains("n_samples") && !j["n_samples"].is_null()) m.n_samples = j["n_samples"].get<int>();
  if (j.contains("cluster_sizes") && !j["cluster_sizes"].is_null())
    m.cluster_sizes = j["cluster_sizes"].get<std::vector<int>>();
  if (j.contains("silhouette_score") && !j["silhouette_score"].is_null())
    m.silhouette_score = j["silhouette_score"].get<float>();
  if (j.contains("inertia") && !j["inertia"].is_null()) m.inertia = j["inertia"].get<float>();
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
  j = {{"cost_bias_min", c.cost_bias_min}, {"cost_bias_max", c.cost_bias_max}};
}

void from_json(const json& j, RoutingConfig& c) {
  c.cost_bias_min = j.value("cost_bias_min", 0.0f);
  c.cost_bias_max = j.value("cost_bias_max", 1.0f);
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

  // Cluster centers - validate structure before accessing
  if (!j.contains("cluster_centers")) {
    throw std::invalid_argument("Missing required field: cluster_centers");
  }

  const auto& centers_json = j.at("cluster_centers");
  if (!centers_json.is_array() || centers_json.empty()) {
    throw std::invalid_argument("cluster_centers must be a non-empty array");
  }

  // Validate n_clusters is positive before casting
  if (checkpoint.clustering.n_clusters <= 0) {
    throw std::invalid_argument(
        std::format("n_clusters must be positive, got {}", checkpoint.clustering.n_clusters));
  }

  auto n_clusters = static_cast<size_t>(checkpoint.clustering.n_clusters);
  if (centers_json.size() != n_clusters) {
    throw std::invalid_argument(std::format("cluster_centers has {} rows but n_clusters is {}",
                                            centers_json.size(), n_clusters));
  }

  // Validate first row exists and is valid array
  if (!centers_json[0].is_array() || centers_json[0].empty()) {
    throw std::invalid_argument("cluster_centers[0] must be a non-empty array");
  }

  size_t feature_dim = centers_json[0].size();

  // Validate all rows have same dimension
  for (size_t i = 0; i < n_clusters; ++i) {
    if (!centers_json[i].is_array()) {
      throw std::invalid_argument(std::format("cluster_centers[{}] is not an array", i));
    }
    if (centers_json[i].size() != feature_dim) {
      throw std::invalid_argument(std::format("cluster_centers[{}] has {} columns but expected {}",
                                              i, centers_json[i].size(), feature_dim));
    }
  }

  if (checkpoint.embedding.dtype == "float64") {
    EmbeddingMatrixT<double> centers(n_clusters, feature_dim);
    for (size_t i = 0; i < n_clusters; ++i) {
      for (size_t col = 0; col < feature_dim; ++col) {
        centers(i, col) = centers_json[i][col].get<double>();
      }
    }
    checkpoint.cluster_centers = std::move(centers);
  } else {
    EmbeddingMatrixT<float> centers(n_clusters, feature_dim);
    for (size_t i = 0; i < n_clusters; ++i) {
      for (size_t col = 0; col < feature_dim; ++col) {
        centers(i, col) = centers_json[i][col].get<float>();
      }
    }
    checkpoint.cluster_centers = std::move(centers);
  }

  checkpoint.validate();

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
        for (size_t i = 0; i < centers.rows(); ++i) {
          json row = json::array();
          for (size_t col = 0; col < centers.cols(); ++col) {
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
// MessagePack File I/O
// ============================================================================

NordlysCheckpoint NordlysCheckpoint::from_msgpack(const std::string& path) {
  NORDLYS_ZONE;

#ifdef _WIN32
  // Windows: use standard file I/O
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error(std::format("Failed to open msgpack file: {}", path));
  }

  auto file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::string data(static_cast<size_t>(file_size), '\0');
  if (!file.read(data.data(), file_size)) {
    throw std::runtime_error(std::format("Failed to read msgpack file: {}", path));
  }

  return from_msgpack_string(data);
#else
  // Unix: use mmap for better performance
  int fd = open(path.c_str(), O_RDONLY);
  if (fd == -1) {
    throw std::runtime_error(std::format("Failed to open msgpack file: {}", path));
  }

  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    close(fd);
    throw std::runtime_error(std::format("Failed to stat msgpack file: {}", path));
  }
  size_t file_size = static_cast<size_t>(sb.st_size);

  void* mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (mapped == MAP_FAILED) {
    close(fd);
    throw std::runtime_error(std::format("Failed to mmap msgpack file: {}", path));
  }

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
#endif
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

  // Cluster centers (binary blob) - validate required fields
  if (!map.contains("cluster_centers")) {
    throw std::invalid_argument("Missing required field: cluster_centers");
  }

  auto centers_map = map.at("cluster_centers").as<std::map<std::string, msgpack::object>>();
  if (!centers_map.contains("rows") || !centers_map.contains("cols")
      || !centers_map.contains("data")) {
    throw std::invalid_argument("cluster_centers must contain rows, cols, and data fields");
  }

  int n_clusters_int = centers_map.at("rows").as<int>();
  int feature_dim_int = centers_map.at("cols").as<int>();

  // Validate dimensions are positive
  if (n_clusters_int <= 0 || feature_dim_int <= 0) {
    throw std::invalid_argument(
        std::format("cluster_centers dimensions must be positive: rows={}, cols={}", n_clusters_int,
                    feature_dim_int));
  }

  // Validate n_clusters matches clustering config
  if (n_clusters_int != checkpoint.clustering.n_clusters) {
    throw std::invalid_argument(
        std::format("cluster_centers rows ({}) does not match n_clusters ({})", n_clusters_int,
                    checkpoint.clustering.n_clusters));
  }

  size_t n_clusters = static_cast<size_t>(n_clusters_int);
  size_t feature_dim = static_cast<size_t>(feature_dim_int);

  // Extract binary data (BIN type only)
  const auto& data_obj = centers_map.at("data");
  if (data_obj.type != msgpack::type::BIN) {
    throw std::invalid_argument("cluster_centers data must be BIN type");
  }
  std::string centers_bytes(data_obj.via.bin.ptr, data_obj.via.bin.size);

  uint64_t total_elements = static_cast<uint64_t>(n_clusters) * static_cast<uint64_t>(feature_dim);

  if (checkpoint.embedding.dtype == "float64") {
    // Check for overflow before computing expected_size
    if (total_elements > SIZE_MAX / sizeof(double)) {
      throw std::invalid_argument(std::format(
          "cluster_centers dimensions too large: {}x{} would overflow", n_clusters, feature_dim));
    }
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
    // Check for overflow before computing expected_size
    if (total_elements > SIZE_MAX / sizeof(float)) {
      throw std::invalid_argument(std::format(
          "cluster_centers dimensions too large: {}x{} would overflow", n_clusters, feature_dim));
    }
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
  pk.pack_map(2);
  pk.pack("cost_bias_min");
  pk.pack(routing.cost_bias_min);
  pk.pack("cost_bias_max");
  pk.pack(routing.cost_bias_max);

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
        if (data_size > std::numeric_limits<uint32_t>::max()) {
          throw std::overflow_error("Matrix data size exceeds uint32_t max for msgpack");
        }
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

        if (static_cast<int>(centers.rows()) != clustering.n_clusters) {
          throw std::invalid_argument(
              std::format("Cluster centers rows ({}) does not match n_clusters ({})",
                          centers.rows(), clustering.n_clusters));
        }

        if (centers.cols() == 0) {
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
