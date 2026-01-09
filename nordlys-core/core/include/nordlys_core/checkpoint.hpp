#pragma once
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "cluster.hpp"
#include "scorer.hpp"

// Version for format evolution
inline constexpr const char* CHECKPOINT_VERSION = "2.0";

// Training metrics (all fields optional for partial state)
struct TrainingMetrics {
  std::optional<int> n_samples;
  std::optional<std::vector<int>> cluster_sizes;
  std::optional<float> silhouette_score;
  std::optional<float> inertia;
};

// Embedding configuration
struct EmbeddingConfig {
  std::string model;  // e.g., "sentence-transformers/all-MiniLM-L6-v2"
  std::string dtype;  // "float32" or "float64" (single source of truth)
  bool trust_remote_code = false;
};

// Clustering hyperparameters (full config for reproducibility)
struct ClusteringConfig {
  int n_clusters;
  int random_state = 42;
  int max_iter = 300;
  int n_init = 10;
  std::string algorithm = "lloyd";
  std::string normalization = "l2";
};

// Routing hyperparameters
struct RoutingConfig {
  float cost_bias_min = 0.0f;
  float cost_bias_max = 1.0f;
  float default_cost_bias = 0.5f;
  int max_alternatives = 5;
};

// Cluster centers (zero-copy variant)
using ClusterCenters = std::variant<EmbeddingMatrixT<float>, EmbeddingMatrixT<double>>;

// Optimized checkpoint structure
struct NordlysCheckpoint {
  std::string version = CHECKPOINT_VERSION;

  // Core data (required for routing)
  ClusterCenters cluster_centers;  // Shape: (n_clusters, feature_dim)
  std::vector<ModelFeatures> models;

  // Configuration (required)
  EmbeddingConfig embedding;
  ClusteringConfig clustering;
  RoutingConfig routing;

  // Training metrics (partial/optional)
  TrainingMetrics metrics;

  // Serialization (zero-copy via mmap)
  [[nodiscard]] static NordlysCheckpoint from_json(const std::string& path);
  [[nodiscard]] static NordlysCheckpoint from_json_string(const std::string& json_str);
  [[nodiscard]] static NordlysCheckpoint from_msgpack(const std::string& path);
  [[nodiscard]] static NordlysCheckpoint from_msgpack_string(const std::string& data);

  void to_json(const std::string& path) const;
  [[nodiscard]] std::string to_json_string() const;
  void to_msgpack(const std::string& path) const;
  [[nodiscard]] std::string to_msgpack_string() const;

  void validate() const;

  // Computed properties (not stored)
  [[nodiscard]] int n_clusters() const {
    return std::visit([](const auto& centers) { return static_cast<int>(centers.rows()); },
                      cluster_centers);
  }

  [[nodiscard]] int feature_dim() const {
    return std::visit([](const auto& centers) { return static_cast<int>(centers.cols()); },
                      cluster_centers);
  }

  // Convenience accessors (aliases to nested fields)
  [[nodiscard]] const std::string& dtype() const { return embedding.dtype; }
  [[nodiscard]] const std::string& embedding_model() const { return embedding.model; }
  [[nodiscard]] int random_state() const { return clustering.random_state; }
  [[nodiscard]] bool allow_trust_remote_code() const { return embedding.trust_remote_code; }
  [[nodiscard]] std::optional<float> silhouette_score() const { return metrics.silhouette_score; }
};
