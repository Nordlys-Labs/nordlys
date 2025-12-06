#pragma once
#include <algorithm>
#include <format>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "cluster.hpp"
#include "profile.hpp"
#include "result.hpp"
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
  [[nodiscard]] static Result<Router, std::string> from_file(const std::string& profile_path) noexcept;
  [[nodiscard]] static Result<Router, std::string> from_json_string(const std::string& json_str) noexcept;
  [[nodiscard]] static Result<Router, std::string> from_binary(const std::string& path) noexcept;

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

  // Thread-safety for lazy initialization (unique_ptr to make Router movable)
  std::unique_ptr<std::once_flag> init_flag_float_ = std::make_unique<std::once_flag>();
  std::unique_ptr<std::once_flag> init_flag_double_ = std::make_unique<std::once_flag>();

  ModelScorer scorer_;
  RouterProfile profile_;
  int embedding_dim_ = 0;
};

// Template implementation inline
template<typename Scalar>
ClusterEngineT<Scalar>& Router::get_cluster_engine() {
  if constexpr (std::is_same_v<Scalar, float>) {
    std::call_once(*init_flag_float_, [this]() {
      cluster_engine_float_.emplace();
      // Convert centroids to float (they're already float in profile)
      cluster_engine_float_->load_centroids(profile_.cluster_centers);
    });
    return *cluster_engine_float_;
  } else if constexpr (std::is_same_v<Scalar, double>) {
    std::call_once(*init_flag_double_, [this]() {
      cluster_engine_double_.emplace();
      // Convert centroids from float to double
      EmbeddingMatrixT<double> centers_double = profile_.cluster_centers.cast<double>();
      cluster_engine_double_->load_centroids(centers_double);
    });
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
    throw std::invalid_argument(
      std::format("Embedding dimension mismatch: expected {} but got {}",
                  embedding_dim_, embedding_size)
    );
  }

  auto& engine = get_cluster_engine<Scalar>();

  EmbeddingVectorT<Scalar> embedding = Eigen::Map<const EmbeddingVectorT<Scalar>>(
      embedding_data, static_cast<Eigen::Index>(embedding_size));

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

    // Extract alternatives using ranges: drop first, take N, extract model_id
    if (alt_count > 0) {
      auto alternatives_view = scores
        | std::views::drop(1)
        | std::views::take(alt_count)
        | std::views::transform(&ModelScore::model_id);

      response.alternatives.assign(alternatives_view.begin(), alternatives_view.end());
    }
  }

   return response;
}
