#pragma once
#include <format>
#include <mutex>
#include <ranges>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#  include <omp.h>
#endif

#include "checkpoint.hpp"
#include "cluster.hpp"
#include "result.hpp"
#include "scorer.hpp"
#include "tracy.hpp"

inline void init_threading() {
#ifdef _OPENMP
  static std::once_flag flag;
  std::call_once(flag, [] {
    omp_set_dynamic(0);
  });
#endif
}

inline void set_num_threads(int n) {
#ifdef _OPENMP
  omp_set_num_threads(n);
#else
  (void)n;
#endif
}

[[nodiscard]] inline int get_num_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

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
    init_threading();

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

  std::vector<RouteResult<Scalar>> route_batch(const Scalar* data, size_t count, size_t dim,
                                                float cost_bias = 0.0f,
                                                const std::vector<std::string>& models = {}) {
    NORDLYS_ZONE_N("Nordlys::route_batch");
    if (dim != static_cast<size_t>(dim_)) {
      throw std::invalid_argument(std::format("dimension mismatch: {} vs {}", dim_, dim));
    }

    auto assignments = engine_.assign_batch(data, count, dim);
    std::vector<RouteResult<Scalar>> results(count);
    bool has_invalid = false;
    auto n = static_cast<ptrdiff_t>(count);

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) reduction(||:has_invalid)
#endif
    for (ptrdiff_t i = 0; i < n; ++i) {
      const auto& [cid, dist] = assignments[static_cast<size_t>(i)];
      if (cid < 0) {
        has_invalid = true;
        continue;
      }

      auto scores = scorer_.score_models(cid, cost_bias, models);

      results[static_cast<size_t>(i)] = RouteResult<Scalar>{
        .selected_model = scores.empty() ? std::string{} : scores[0].model_id,
        .alternatives = [&]() -> std::vector<std::string> {
          if (scores.size() <= 1) return {};
          auto alts = scores | std::views::drop(1) | std::views::transform(&ModelScore::model_id);
          return {alts.begin(), alts.end()};
        }(),
        .cluster_id = cid,
        .cluster_distance = dist
      };
    }

    if (has_invalid) {
      throw std::runtime_error("no valid cluster");
    }

    return results;
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

    const auto& centers = std::get<EmbeddingMatrix<Scalar>>(checkpoint_.cluster_centers);
    dim_ = static_cast<int>(centers.cols());
    engine_.load_centroids(centers);

    scorer_.load_models(checkpoint_.models);
    scorer_.set_lambda_params(checkpoint_.routing.cost_bias_min, checkpoint_.routing.cost_bias_max);
  }

  ClusterEngine<Scalar> engine_;
  ModelScorer scorer_;
  NordlysCheckpoint checkpoint_;
  int dim_ = 0;
};

using Nordlys32 = Nordlys<float>;
using Nordlys64 = Nordlys<double>;
