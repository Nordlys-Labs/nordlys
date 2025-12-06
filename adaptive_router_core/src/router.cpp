#include "router.hpp"

#include <algorithm>
#include <ranges>
#include <stdexcept>

void Router::initialize(const RouterProfile& prof) {
  profile_ = prof;
  embedding_dim_ = static_cast<int>(prof.cluster_centers.cols());

  // Don't initialize cluster engines here - they're lazy-initialized on first use
  scorer_.load_models(prof.models);
  scorer_.set_lambda_params(prof.metadata.routing.lambda_min, prof.metadata.routing.lambda_max);
}

std::expected<Router, std::string> Router::from_file(const std::string& profile_path) noexcept {
  try {
    Router router;
    auto profile = RouterProfile::from_json(profile_path);
    router.initialize(profile);
    return router;
  } catch (const std::exception& e) {
    return std::unexpected(e.what());
  }
}

std::expected<Router, std::string> Router::from_json_string(const std::string& json_str) noexcept {
  try {
    Router router;
    auto profile = RouterProfile::from_json_string(json_str);
    router.initialize(profile);
    return router;
  } catch (const std::exception& e) {
    return std::unexpected(e.what());
  }
}

std::expected<Router, std::string> Router::from_binary(const std::string& path) noexcept {
  try {
    Router router;
    auto profile = RouterProfile::from_binary(path);
    router.initialize(profile);
    return router;
  } catch (const std::exception& e) {
    return std::unexpected(e.what());
  }
}

std::vector<std::string> Router::get_supported_models() const {
  auto model_ids = profile_.models
    | std::views::transform([](const auto& m) { return m.model_id; });
  return {model_ids.begin(), model_ids.end()};
}

int Router::get_n_clusters() const noexcept {
  // Return from whichever engine is initialized, or from profile
  if (cluster_engine_float_) {
    return cluster_engine_float_->get_n_clusters();
  }
  if (cluster_engine_double_) {
    return cluster_engine_double_->get_n_clusters();
  }
  return static_cast<int>(profile_.cluster_centers.rows());
}

int Router::get_embedding_dim() const noexcept {
  return embedding_dim_;
}
