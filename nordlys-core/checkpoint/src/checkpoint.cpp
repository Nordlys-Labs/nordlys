#ifdef _WIN32
#  include <io.h>
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <format>
#include <fstream>
#include <limits>
#include <msgpack.hpp>
#include <nlohmann/json.hpp>
#include <nordlys/checkpoint/cache.hpp>
#include <nordlys/checkpoint/checkpoint.hpp>
#include <ranges>
#include <stdexcept>

using json = nlohmann::json;

namespace {
  nordlys::LruCache<NordlysCheckpoint>& checkpoint_cache() {
    static nordlys::LruCache<NordlysCheckpoint> cache;
    return cache;
  }

  json parse_json_or_throw(const std::string& raw, const char* field_name) {
    try {
      return json::parse(raw);
    } catch (const json::exception& e) {
      throw std::invalid_argument(
          std::format("Failed to parse {} JSON payload: {}", field_name, e.what()));
    }
  }
}  // namespace

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

void to_json(json& j, const EmbeddingConfig& c) {
  j = {{"model", c.model}, {"trust_remote_code", c.trust_remote_code}};
}

void from_json(const json& j, EmbeddingConfig& c) {
  j.at("model").get_to(c.model);
  c.trust_remote_code = j.value("trust_remote_code", false);
}

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

void to_json(json& j, const ReductionConfig& c) {
  j = {{"kind", c.kind},
       {"config", parse_json_or_throw(c.config_json, "reduction.config")},
       {"state", parse_json_or_throw(c.state_json, "reduction.state")}};
}

void from_json(const json& j, ReductionConfig& c) {
  j.at("kind").get_to(c.kind);
  c.config_json = j.contains("config") ? j.at("config").dump() : "{}";
  c.state_json = j.contains("state") ? j.at("state").dump() : "{}";
}

void to_json(json& j, const ModelFeatures& f) {
  j = {{"model_id", f.model_id}, {"scores", f.scores}};
}

void from_json(const json& j, ModelFeatures& f) {
  j.at("model_id").get_to(f.model_id);
  j.at("scores").get_to(f.scores);
}

NordlysCheckpoint NordlysCheckpoint::from_json(const std::string& path) {
  if (auto cached = checkpoint_cache().get(path)) {
    return *cached;
  }

  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) [[unlikely]] {
    throw std::runtime_error(std::format("Failed to open checkpoint file: {}", path));
  }

  const auto file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::string content(static_cast<size_t>(file_size), '\0');
  if (!file.read(content.data(), file_size)) [[unlikely]] {
    throw std::runtime_error(std::format("Failed to read checkpoint file: {}", path));
  }

  auto checkpoint = std::make_shared<NordlysCheckpoint>(from_json_string(content));
  checkpoint_cache().put(path, checkpoint);
  return *checkpoint;
}

NordlysCheckpoint NordlysCheckpoint::from_json_string(const std::string& json_str) {
  json doc;
  try {
    doc = json::parse(json_str);
  } catch (const json::exception& e) {
    throw std::invalid_argument(std::format("Failed to parse JSON: {}", e.what()));
  }

  NordlysCheckpoint checkpoint;
  doc.at("embedding").get_to(checkpoint.embedding);
  doc.at("clustering").get_to(checkpoint.clustering);
  doc.at("models").get_to(checkpoint.models);

  if (!doc.contains("reduction")) {
    throw std::invalid_argument("Missing required field: reduction");
  }
  if (!doc["reduction"].is_null()) {
    checkpoint.reduction = doc["reduction"].get<ReductionConfig>();
  }

  if (doc.contains("metrics") && !doc["metrics"].is_null()) {
    doc["metrics"].get_to(checkpoint.metrics);
  }

  auto centers_arr = doc.at("cluster_centers");
  if (!centers_arr.is_array()) {
    throw std::invalid_argument("cluster_centers must be an array");
  }

  const auto n_clusters = static_cast<size_t>(checkpoint.clustering.n_clusters);
  std::vector<float> all_values;
  size_t feature_dim = 0;
  size_t row_count = 0;
  for (const auto& row : centers_arr) {
    if (!row.is_array()) {
      throw std::invalid_argument("cluster_centers rows must be arrays");
    }
    const auto col_count = row.size();
    if (row_count == 0) {
      feature_dim = col_count;
      all_values.reserve(n_clusters * feature_dim);
    } else if (col_count != feature_dim) {
      throw std::invalid_argument(
          std::format("cluster_centers row {} has {} columns but expected {} (from first row)",
                      row_count, col_count, feature_dim));
    }
    for (const auto& value : row) {
      all_values.push_back(value.get<float>());
    }
    ++row_count;
  }

  if (row_count != n_clusters) {
    throw std::invalid_argument(
        std::format("cluster_centers has {} rows but n_clusters is {}", row_count, n_clusters));
  }

  EmbeddingMatrix<float> centers(n_clusters, feature_dim);
  std::memcpy(centers.data(), all_values.data(), all_values.size() * sizeof(float));
  checkpoint.cluster_centers = std::move(centers);
  checkpoint.validate();
  return checkpoint;
}

std::string NordlysCheckpoint::to_json_string() const {
  json j;

  j["embedding"] = embedding;
  j["clustering"] = clustering;
  j["reduction"] = reduction ? json(*reduction) : json(nullptr);

  j["models"] = models;

  j["metrics"] = metrics;

  json centers_array = json::array();
  for (size_t i = 0; i < cluster_centers.rows(); ++i) {
    json row = json::array();
    for (size_t col = 0; col < cluster_centers.cols(); ++col) {
      row.push_back(cluster_centers(i, col));
    }
    centers_array.push_back(row);
  }
  j["cluster_centers"] = centers_array;

  return j.dump(2);
}

void NordlysCheckpoint::to_json(const std::string& path) const {
  std::ofstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error(std::format("Failed to open checkpoint file for writing: {}", path));
  }
  file << to_json_string();
}

NordlysCheckpoint NordlysCheckpoint::from_msgpack(const std::string& path) {
  if (auto cached = checkpoint_cache().get(path)) {
    return *cached;
  }

#ifdef _WIN32
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

  auto checkpoint = std::make_shared<NordlysCheckpoint>(from_msgpack_string(data));
#else
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

  std::shared_ptr<NordlysCheckpoint> checkpoint;
  try {
    checkpoint = std::make_shared<NordlysCheckpoint>(
        from_msgpack_string(std::string(static_cast<const char*>(mapped), file_size)));
    munmap(mapped, file_size);
    close(fd);
  } catch (...) {
    munmap(mapped, file_size);
    close(fd);
    throw;
  }
#endif

  checkpoint_cache().put(path, checkpoint);
  return *checkpoint;
}

NordlysCheckpoint NordlysCheckpoint::from_msgpack_string(const std::string& data) {
  msgpack::object_handle handle = msgpack::unpack(data.data(), data.size());
  auto map = handle.get().as<std::map<std::string, msgpack::object>>();

  NordlysCheckpoint checkpoint;

  auto emb = map.at("embedding").as<std::map<std::string, msgpack::object>>();
  checkpoint.embedding.model = emb.at("model").as<std::string>();
  checkpoint.embedding.trust_remote_code
      = emb.contains("trust_remote_code") ? emb.at("trust_remote_code").as<bool>() : false;

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

  if (!map.contains("reduction")) {
    throw std::invalid_argument("Missing required field: reduction");
  }
  if (map.at("reduction").type != msgpack::type::NIL) {
    auto reduction_map = map.at("reduction").as<std::map<std::string, msgpack::object>>();
    ReductionConfig reduction;
    reduction.kind = reduction_map.at("kind").as<std::string>();
    reduction.config_json
        = reduction_map.contains("config") ? reduction_map.at("config").as<std::string>() : "{}";
    reduction.state_json
        = reduction_map.contains("state") ? reduction_map.at("state").as<std::string>() : "{}";
    checkpoint.reduction = std::move(reduction);
  }

  auto models_arr = map.at("models").as<std::vector<msgpack::object>>();
  checkpoint.models.reserve(models_arr.size());
  for (const auto& model_obj : models_arr) {
    auto model_map = model_obj.as<std::map<std::string, msgpack::object>>();
    ModelFeatures model;
    model.model_id = model_map.at("model_id").as<std::string>();
    model.scores = model_map.at("scores").as<std::vector<float>>();
    checkpoint.models.push_back(std::move(model));
  }

  if (map.contains("metrics")) {
    auto met = map.at("metrics").as<std::map<std::string, msgpack::object>>();
    if (met.contains("n_samples")) checkpoint.metrics.n_samples = met.at("n_samples").as<int>();
    if (met.contains("cluster_sizes"))
      checkpoint.metrics.cluster_sizes = met.at("cluster_sizes").as<std::vector<int>>();
    if (met.contains("silhouette_score"))
      checkpoint.metrics.silhouette_score = met.at("silhouette_score").as<float>();
    if (met.contains("inertia")) checkpoint.metrics.inertia = met.at("inertia").as<float>();
  }

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

  if (n_clusters_int <= 0 || feature_dim_int <= 0) {
    throw std::invalid_argument(
        std::format("cluster_centers dimensions must be positive: rows={}, cols={}", n_clusters_int,
                    feature_dim_int));
  }

  if (n_clusters_int != checkpoint.clustering.n_clusters) {
    throw std::invalid_argument(
        std::format("cluster_centers rows ({}) does not match n_clusters ({})", n_clusters_int,
                    checkpoint.clustering.n_clusters));
  }

  size_t n_clusters = static_cast<size_t>(n_clusters_int);
  size_t feature_dim = static_cast<size_t>(feature_dim_int);

  const auto& data_obj = centers_map.at("data");
  if (data_obj.type != msgpack::type::BIN) {
    throw std::invalid_argument("cluster_centers data must be BIN type");
  }
  std::string centers_bytes(data_obj.via.bin.ptr, data_obj.via.bin.size);

  uint64_t total_elements = static_cast<uint64_t>(n_clusters) * static_cast<uint64_t>(feature_dim);

  if (total_elements > SIZE_MAX / sizeof(float)) {
    throw std::invalid_argument(std::format(
        "cluster_centers dimensions too large: {}x{} would overflow", n_clusters, feature_dim));
  }

  size_t expected_size = total_elements * sizeof(float);

  EmbeddingMatrix<float> centers(n_clusters, feature_dim);

  if (centers_bytes.size() != expected_size) {
    throw std::invalid_argument(
        std::format("cluster_centers data size mismatch: expected {} bytes (float32 only), got {}",
                    expected_size, centers_bytes.size()));
  }

  std::memcpy(centers.data(), centers_bytes.data(), expected_size);

  checkpoint.cluster_centers = std::move(centers);

  return checkpoint;
}

std::string NordlysCheckpoint::to_msgpack_string() const {
  msgpack::sbuffer buffer;
  msgpack::packer<msgpack::sbuffer> pk(&buffer);

  pk.pack_map(6);

  pk.pack("embedding");
  pk.pack_map(2);
  pk.pack("model");
  pk.pack(embedding.model);
  pk.pack("trust_remote_code");
  pk.pack(embedding.trust_remote_code);

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

  pk.pack("reduction");
  if (reduction) {
    pk.pack_map(3);
    pk.pack("kind");
    pk.pack(reduction->kind);
    pk.pack("config");
    pk.pack(reduction->config_json);
    pk.pack("state");
    pk.pack(reduction->state_json);
  } else {
    pk.pack_nil();
  }

  pk.pack("models");
  pk.pack_array(static_cast<uint32_t>(models.size()));
  for (const auto& model : models) {
    pk.pack_map(2);
    pk.pack("model_id");
    pk.pack(model.model_id);
    pk.pack("scores");
    pk.pack(model.scores);
  }

  pk.pack("cluster_centers");
  pk.pack_map(3);
  pk.pack("rows");
  pk.pack(static_cast<int>(cluster_centers.rows()));
  pk.pack("cols");
  pk.pack(static_cast<int>(cluster_centers.cols()));
  pk.pack("data");
  size_t data_size = static_cast<size_t>(cluster_centers.rows())
                     * static_cast<size_t>(cluster_centers.cols()) * sizeof(float);
  if (data_size > std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error("Matrix data size exceeds uint32_t max for msgpack");
  }
  pk.pack_bin(static_cast<uint32_t>(data_size));
  pk.pack_bin_body(reinterpret_cast<const char*>(cluster_centers.data()),
                   static_cast<uint32_t>(data_size));

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

void NordlysCheckpoint::validate() const {
  if (clustering.n_clusters <= 0) {
    throw std::invalid_argument(
        std::format("n_clusters must be positive, got {}", clustering.n_clusters));
  }

  if (static_cast<int>(cluster_centers.rows()) != clustering.n_clusters) {
    throw std::invalid_argument(
        std::format("Cluster centers rows ({}) does not match n_clusters ({})",
                    cluster_centers.rows(), clustering.n_clusters));
  }

  if (cluster_centers.cols() == 0) {
    throw std::invalid_argument(
        std::format("feature_dim must be positive, got {}", cluster_centers.cols()));
  }

  if (models.empty()) {
    throw std::invalid_argument("models array cannot be empty");
  }

  if (reduction) {
    if (reduction->kind.empty()) {
      throw std::invalid_argument("reduction.kind must be non-empty");
    }
    parse_json_or_throw(reduction->config_json, "reduction.config");
    parse_json_or_throw(reduction->state_json, "reduction.state");
  }

  for (size_t i = 0; i < models.size(); ++i) {
    const auto& model = models[i];

    if (model.scores.size() != static_cast<size_t>(clustering.n_clusters)) {
      throw std::invalid_argument(
          std::format("Model {} scores size ({}) does not match n_clusters ({})", i,
                      model.scores.size(), clustering.n_clusters));
    }

    for (size_t j = 0; j < model.scores.size(); ++j) {
      float score = model.scores[j];
      if (score < 0.0f || score > 1.0f) {
        throw std::invalid_argument(std::format(
            "Model {} score[{}] ({}) must be in range [0.0, 1.0]", i, j, score));
      }
    }
  }
}
