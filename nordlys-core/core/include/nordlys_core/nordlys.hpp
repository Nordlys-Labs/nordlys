#pragma once
#include <format>
#include <ranges>
#include <stdexcept>
#include <string>
#include <vector>

#include "checkpoint.hpp"
#include "cluster.hpp"
#include "result.hpp"
#include "scorer.hpp"
#include "tracy.hpp"

template <typename Scalar = float> struct RouteResult {
  std::string selected_model;
  std::vector<std::string> alternatives;
  int cluster_id;
  Scalar cluster_distance;
};

template <typename Scalar = float> class Nordlys {
public:
  using value_type = Scalar;

  static Result<Nordlys, std::string> from_checkpoint(NordlysCheckpoint checkpoint) noexcept {
    NORDLYS_ZONE_N("Nordlys::from_checkpoint");
    if constexpr (std::is_same_v<Scalar, float>) {
      if (checkpoint.dtype() != "float32") {
        return Unexpected("Nordlys<float> requires float32 checkpoint, but checkpoint dtype is "
                          + checkpoint.dtype());
      }
    } else if constexpr (std::is_same_v<Scalar, double>) {
      if (checkpoint.dtype() != "float64") {
        return Unexpected("Nordlys<double> requires float64 checkpoint, but checkpoint dtype is "
                          + checkpoint.dtype());
      }
    }

    try {
      Nordlys engine;
      engine.init(std::move(checkpoint));
      return engine;
    } catch (const std::exception& e) {
      return Unexpected(std::string(e.what()));
    }
  }

  Nordlys() = default;
  Nordlys(Nordlys&&) = default;
  Nordlys& operator=(Nordlys&&) = default;
  Nordlys(const Nordlys&) = delete;
  Nordlys& operator=(const Nordlys&) = delete;

  RouteResult<Scalar> route(const Scalar* data, size_t size, float cost_bias = 0.0f,
                            const std::vector<std::string>& models = {}) {
    NORDLYS_ZONE_N("Nordlys::route");
    if (size != static_cast<size_t>(dim_)) {
      throw std::invalid_argument(std::format("dimension mismatch: {} vs {}", dim_, size));
    }

    auto [cid, dist] = engine_.assign(data, size);

    if (cid < 0) throw std::runtime_error("no valid cluster");

    auto scores = scorer_.score_models(cid, cost_bias, models);

    RouteResult<Scalar> resp{.selected_model = scores.empty() ? "" : scores[0].model_id,
                             .alternatives = {},
                             .cluster_id = cid,
                             .cluster_distance = dist};

    if (scores.size() > 1) {
      auto alts = scores | std::views::drop(1) | std::views::transform(&ModelScore::model_id);
      resp.alternatives.assign(alts.begin(), alts.end());
    }
    return resp;
  }

  std::vector<std::string> get_supported_models() const {
    auto ids = checkpoint_.models | std::views::transform(&ModelFeatures::model_id);
    return {ids.begin(), ids.end()};
  }

  int get_n_clusters() const { return engine_.get_n_clusters(); }
  int get_embedding_dim() const { return dim_; }

private:
  void init(NordlysCheckpoint checkpoint) {
    NORDLYS_ZONE_N("Nordlys::init");
    checkpoint_ = std::move(checkpoint);

    const auto& centers = std::get<EmbeddingMatrixT<Scalar>>(checkpoint_.cluster_centers);
    dim_ = static_cast<int>(centers.cols());
    engine_.load_centroids(centers);

    scorer_.load_models(checkpoint_.models);
    scorer_.set_lambda_params(checkpoint_.routing.cost_bias_min, checkpoint_.routing.cost_bias_max);
  }

  ClusterEngineT<Scalar> engine_;
  ModelScorer scorer_;
  NordlysCheckpoint checkpoint_;
  int dim_ = 0;
};

using Nordlys32 = Nordlys<float>;
using Nordlys64 = Nordlys<double>;
