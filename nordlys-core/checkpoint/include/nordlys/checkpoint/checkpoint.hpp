#pragma once
#include <optional>
#include <string>
#include <vector>

#include <nordlys/common/matrix.hpp>
#include <nordlys/scoring/scorer.hpp>

inline constexpr const char* CHECKPOINT_VERSION = "2.0";

struct TrainingMetrics {
  std::optional<int> n_samples;
  std::optional<std::vector<int>> cluster_sizes;
  std::optional<float> silhouette_score;
  std::optional<float> inertia;
};

struct EmbeddingConfig {
  std::string model;
  bool trust_remote_code;
};

struct ClusteringConfig {
  int n_clusters;
  int random_state;
  int max_iter;
  int n_init;
  std::string algorithm;
  std::string normalization;
};

struct NordlysCheckpoint {
  std::string version;

  EmbeddingMatrix<float> cluster_centers;
  std::vector<ModelFeatures> models;

  EmbeddingConfig embedding;
  ClusteringConfig clustering;

  TrainingMetrics metrics;

  [[nodiscard]] static NordlysCheckpoint from_json(const std::string& path);
  [[nodiscard]] static NordlysCheckpoint from_json_string(const std::string& json_str);
  [[nodiscard]] static NordlysCheckpoint from_msgpack(const std::string& path);
  [[nodiscard]] static NordlysCheckpoint from_msgpack_string(const std::string& data);

  void to_json(const std::string& path) const;
  [[nodiscard]] std::string to_json_string() const;
  void to_msgpack(const std::string& path) const;
  [[nodiscard]] std::string to_msgpack_string() const;

  void validate() const;

  [[nodiscard]] int n_clusters() const { return static_cast<int>(cluster_centers.rows()); }

  [[nodiscard]] int feature_dim() const { return static_cast<int>(cluster_centers.cols()); }

  [[nodiscard]] const std::string& embedding_model() const { return embedding.model; }
  [[nodiscard]] int random_state() const { return clustering.random_state; }
  [[nodiscard]] bool allow_trust_remote_code() const { return embedding.trust_remote_code; }
  [[nodiscard]] float silhouette_score() const { return metrics.silhouette_score.value_or(-1.0f); }
};
