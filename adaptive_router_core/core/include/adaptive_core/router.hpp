#pragma once
#include <format>
#include <ranges>
#include <string>
#include <vector>

#include "cluster.hpp"
#include "profile.hpp"
#include "result.hpp"
#include "scorer.hpp"

template<typename Scalar = float>
struct RouteResponseT {
  std::string selected_model;
  std::vector<std::string> alternatives;
  int cluster_id;
  Scalar cluster_distance;
};

template<typename Scalar = float>
class RouterT {
public:
  using value_type = Scalar;  // Enable type detection in std::visit

  static Result<RouterT, std::string> from_profile(RouterProfile profile) noexcept {
    // Validate dtype compatibility before initialization
    if constexpr (std::is_same_v<Scalar, float>) {
      if (!profile.is_float32()) {
        return Unexpected("RouterT<float> requires float32 profile, but profile dtype is " +
                         profile.metadata.dtype);
      }
    } else if constexpr (std::is_same_v<Scalar, double>) {
      if (!profile.is_float64()) {
        return Unexpected("RouterT<double> requires float64 profile, but profile dtype is " +
                         profile.metadata.dtype);
      }
    }

    try {
      RouterT r;
      r.init(std::move(profile));
      return r;
    } catch (const std::exception& e) {
      return Unexpected(std::string(e.what()));
    }
  }

  RouterT() = default;
  RouterT(RouterT&&) = default;
  RouterT& operator=(RouterT&&) = default;
  RouterT(const RouterT&) = delete;
  RouterT& operator=(const RouterT&) = delete;

  RouteResponseT<Scalar> route(const Scalar* data, size_t size, float cost_bias = 0.5f,
                           const std::vector<std::string>& models = {}) {
    if (size != static_cast<size_t>(dim_)) {
      throw std::invalid_argument(std::format("dimension mismatch: {} vs {}", dim_, size));
    }

    auto emb = Eigen::Map<const EmbeddingVectorT<Scalar>>(data, dim_);
    auto [cid, dist] = engine_.assign(emb);

    if (cid < 0) throw std::runtime_error("no valid cluster");

    auto scores = scorer_.score_models(cid, cost_bias, models);

    RouteResponseT<Scalar> resp{
      .selected_model = scores.empty() ? "" : scores[0].model_id,
      .alternatives = {},
      .cluster_id = cid,
      .cluster_distance = dist
    };

    if (scores.size() > 1) {
      size_t n = std::min(scores.size() - 1, static_cast<size_t>(profile_.metadata.routing.max_alternatives));
      auto alts = scores | std::views::drop(1) | std::views::take(n)
                        | std::views::transform(&ModelScore::model_id);
      resp.alternatives.assign(alts.begin(), alts.end());
    }
    return resp;
  }

  std::vector<std::string> get_supported_models() const {
    auto ids = profile_.models | std::views::transform(&ModelFeatures::model_id);
    return {ids.begin(), ids.end()};
  }

  int get_n_clusters() const { return engine_.get_n_clusters(); }
  int get_embedding_dim() const { return dim_; }

private:
  void init(RouterProfile profile) {
    profile_ = std::move(profile);

    // Get centers with correct type from variant
    const auto& centers = profile_.centers<Scalar>();
    dim_ = static_cast<int>(centers.cols());
    engine_.load_centroids(centers);

    scorer_.load_models(profile_.models);
    scorer_.set_lambda_params(profile_.metadata.routing.lambda_min,
                               profile_.metadata.routing.lambda_max);
  }

  ClusterEngineT<Scalar> engine_;
  ModelScorer scorer_;
  RouterProfile profile_;
  int dim_ = 0;
};

using Router = RouterT<float>;
