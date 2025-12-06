#pragma once
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "cluster.hpp"
#include "profile.hpp"
#include "scorer.hpp"

struct RouteRequest {
  std::span<const float> embedding;
  float cost_bias = 0.5f;
  std::vector<std::string> models;
};

struct RouteResponse {
  std::string selected_model;
  std::vector<std::string> alternatives;
  int cluster_id;
  float cluster_distance;
};

class Router {
public:
  [[nodiscard]] static std::expected<Router, std::string> from_file(const std::string& profile_path) noexcept;
  [[nodiscard]] static std::expected<Router, std::string> from_json_string(const std::string& json_str) noexcept;
  [[nodiscard]] static std::expected<Router, std::string> from_binary(const std::string& path) noexcept;

  Router() = default;
  ~Router() = default;

  Router(Router&&) = default;
  Router& operator=(Router&&) = default;
  Router(const Router&) = delete;
  Router& operator=(const Router&) = delete;

  // Main routing API - templated to accept any floating point type
  template<typename Scalar>
  [[nodiscard]] RouteResponse route(const Scalar* embedding_data, size_t embedding_size, float cost_bias = 0.5f,
                                   const std::vector<std::string>& models = {});

  [[nodiscard]] std::vector<std::string> get_supported_models() const;
  [[nodiscard]] int get_n_clusters() const noexcept;
  [[nodiscard]] int get_embedding_dim() const noexcept;

private:
  void initialize(const RouterProfile& prof);

  // Ensure cluster engine for given type is initialized
  template<typename Scalar>
  ClusterEngineT<Scalar>& get_cluster_engine();

  // Lazy-initialized engines - only created when first used
  std::optional<ClusterEngineT<float>> cluster_engine_float_;
  std::optional<ClusterEngineT<double>> cluster_engine_double_;

  ModelScorer scorer_;
  RouterProfile profile_;
  int embedding_dim_ = 0;
};

// Template implementation inline
template<typename Scalar>
ClusterEngineT<Scalar>& Router::get_cluster_engine() {
  if constexpr (std::is_same_v<Scalar, float>) {
    if (!cluster_engine_float_) {
      cluster_engine_float_.emplace();
      // Convert centroids to float (they're already float in profile)
      cluster_engine_float_->load_centroids(profile_.cluster_centers);
    }
    return *cluster_engine_float_;
  } else if constexpr (std::is_same_v<Scalar, double>) {
    if (!cluster_engine_double_) {
      cluster_engine_double_.emplace();
      // Convert centroids from float to double
      EmbeddingMatrixT<double> centers_double = profile_.cluster_centers.cast<double>();
      cluster_engine_double_->load_centroids(centers_double);
    }
    return *cluster_engine_double_;
  } else {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>,
                  "Only float and double are supported");
  }
}

template<typename Scalar>
RouteResponse Router::route(const Scalar* embedding_data, size_t embedding_size, float cost_bias,
                           const std::vector<std::string>& models) {
  if (embedding_size != static_cast<size_t>(embedding_dim_)) {
    throw std::invalid_argument("Embedding dimension mismatch: expected "
                                + std::to_string(embedding_dim_) + " but got "
                                + std::to_string(embedding_size));
  }

  auto& engine = get_cluster_engine<Scalar>();

  EmbeddingVectorT<Scalar> embedding = Eigen::Map<const EmbeddingVectorT<Scalar>>(
      embedding_data, embedding_size);

   auto [cluster_id, distance] = engine.assign(embedding);

   // Validate cluster assignment
   if (cluster_id < 0) {
     throw std::runtime_error(
         "No valid cluster found for embedding; check router profile configuration");
   }

   // Score models for this cluster
   auto scores = scorer_.score_models(cluster_id, cost_bias, models);

  RouteResponse response;
  response.cluster_id = cluster_id;
  response.cluster_distance = static_cast<float>(distance);

  if (!scores.empty()) {
    response.selected_model = scores[0].model_id;

    int max_alt = profile_.metadata.routing.max_alternatives;
    int alt_count = std::max(0, std::min<int>(static_cast<int>(scores.size()) - 1, max_alt));
    if (alt_count > 0) {
      response.alternatives.reserve(static_cast<std::size_t>(alt_count));
    }

    for (int i = 1; i <= alt_count; ++i) {
      response.alternatives.push_back(scores[static_cast<std::size_t>(i)].model_id);
    }
  }

   return response;
}
