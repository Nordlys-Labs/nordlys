#include "router.hpp"

#include <algorithm>
#include <ranges>
#include <stdexcept>

void Router::initialize(const RouterProfile& prof) {
  profile_ = prof;
  embedding_dim_ = static_cast<int>(prof.cluster_centers.cols());

  cluster_engine_.load_centroids(prof.cluster_centers);

  scorer_.load_models(prof.models);
  scorer_.set_lambda_params(prof.metadata.routing.lambda_min, prof.metadata.routing.lambda_max);
}

Router Router::from_file(const std::string& profile_path) {
  Router router;
  auto profile = RouterProfile::from_json(profile_path);
  router.initialize(profile);
  return router;
}

Router Router::from_json_string(const std::string& json_str) {
  Router router;
  auto profile = RouterProfile::from_json_string(json_str);
  router.initialize(profile);
  return router;
}

Router Router::from_binary(const std::string& path) {
  Router router;
  auto profile = RouterProfile::from_binary(path);
  router.initialize(profile);
  return router;
}

RouteResponse Router::route(const RouteRequest& request) {
  // Zero-copy: map span to Eigen vector view
  Eigen::Map<const EmbeddingVector> embedding(
    request.embedding.data(),
    static_cast<Eigen::Index>(request.embedding.size())
  );

  return route(embedding, request.cost_bias);
}

RouteResponse Router::route(const float* embedding_data, size_t embedding_size, float cost_bias) {
  // Zero-copy: map raw pointer to Eigen vector view
  Eigen::Map<const EmbeddingVector> embedding(
    embedding_data,
    static_cast<Eigen::Index>(embedding_size)
  );

  return route(embedding, cost_bias);
}

RouteResponse Router::route(const EmbeddingVector& embedding, float cost_bias) {
  // Validate cost_bias range [0.0, 1.0]
  if (cost_bias < 0.0f || cost_bias > 1.0f) {
    throw std::invalid_argument("cost_bias must be in range [0.0, 1.0], got "
                                + std::to_string(cost_bias));
  }

  // Validate embedding dimensions
  if (embedding.size() != embedding_dim_) {
    throw std::invalid_argument("Embedding dimension mismatch: expected "
                                + std::to_string(embedding_dim_) + " but got "
                                + std::to_string(embedding.size()));
  }

  // Assign to cluster
  auto [cluster_id, distance] = cluster_engine_.assign(embedding);

  // Score models for this cluster
  auto scores = scorer_.score_models(cluster_id, cost_bias);

  RouteResponse response;
  response.cluster_id = cluster_id;
  response.cluster_distance = distance;

  if (!scores.empty()) {
    response.selected_model = scores[0].model_id;

    // Add alternatives using modern C++ (skip first, which is selected)
    int max_alt = profile_.metadata.routing.max_alternatives;
    auto alt_count = std::min(static_cast<int>(scores.size()) - 1, max_alt);
    response.alternatives.reserve(static_cast<std::size_t>(alt_count));

    for (int i = 1; i <= alt_count; ++i) {
      response.alternatives.push_back(scores[static_cast<std::size_t>(i)].model_id);
    }
  }

  return response;
}

std::vector<std::string> Router::get_supported_models() const {
  auto model_ids = profile_.models
    | std::views::transform([](const auto& m) { return m.model_id; });
  return {model_ids.begin(), model_ids.end()};
}

int Router::get_n_clusters() const noexcept { return cluster_engine_.get_n_clusters(); }

int Router::get_embedding_dim() const noexcept { return embedding_dim_; }
