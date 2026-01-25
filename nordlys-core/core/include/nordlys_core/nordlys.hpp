#pragma once
#include <format>
#include <iterator>
#include <mutex>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#ifdef _OPENMP
#  include <omp.h>
#endif

#include "checkpoint.hpp"
#include "cluster.hpp"
#include "device.hpp"
#include "embedding_view.hpp"
#include "result.hpp"
#include "scorer.hpp"

inline void init_threading() {
#ifdef _OPENMP
  static std::once_flag flag;
  std::call_once(flag, [] { omp_set_dynamic(0); });
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

  static Result<Nordlys, std::string> from_checkpoint(NordlysCheckpoint checkpoint,
                                                      Device device = CpuDevice{}) noexcept {
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
      engine.init(std::move(checkpoint), device);
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

  RouteResult<Scalar> route(EmbeddingView<Scalar> view) {
    if (!backend_) [[unlikely]] {
      throw std::runtime_error("Nordlys not initialized; call from_checkpoint() or init()");
    }
    return route_impl(view, std::nullopt);
  }

  RouteResult<Scalar> route(EmbeddingView<Scalar> view,
                            const std::vector<std::string>& model_filter) {
    if (!backend_) [[unlikely]] {
      throw std::runtime_error("Nordlys not initialized; call from_checkpoint() or init()");
    }
    return route_impl(view,
                      model_filter.empty()
                          ? std::nullopt
                          : std::optional<std::reference_wrapper<const std::vector<std::string>>>(model_filter));
  }

  std::vector<RouteResult<Scalar>> route_batch(EmbeddingBatchView<Scalar> view) {
    if (!backend_) [[unlikely]] {
      throw std::runtime_error("Nordlys not initialized; call from_checkpoint() or init()");
    }
    return route_batch_impl(view, std::nullopt);
  }

  std::vector<RouteResult<Scalar>> route_batch(EmbeddingBatchView<Scalar> view,
                                               const std::vector<std::string>& model_filter) {
    if (!backend_) [[unlikely]] {
      throw std::runtime_error("Nordlys not initialized; call from_checkpoint() or init()");
    }
    return route_batch_impl(view,
                            model_filter.empty()
                                ? std::nullopt
                                : std::optional<std::reference_wrapper<const std::vector<std::string>>>(model_filter));
  }

  std::vector<std::string> get_supported_models() const {
    auto ids = checkpoint_.models | std::views::transform(&ModelFeatures::model_id);
    return {ids.begin(), ids.end()};
  }

  size_t get_n_clusters() const {
    if (!backend_) [[unlikely]] {
      throw std::runtime_error("Nordlys not initialized; call from_checkpoint() or init()");
    }
    return backend_->n_clusters();
  }
  size_t get_embedding_dim() const { return dim_; }

private:
  RouteResult<Scalar> route_impl(
      EmbeddingView<Scalar> view,
      std::optional<std::reference_wrapper<const std::vector<std::string>>> model_filter) {
    if (view.dim != dim_) [[unlikely]] {
      throw std::invalid_argument(std::format("dimension mismatch: {} vs {}", dim_, view.dim));
    }

    auto [cid, dist] = backend_->assign(view);

    if (cid < 0) [[unlikely]]
      throw std::runtime_error("no valid cluster");

    auto models = get_models_to_score(model_filter);
    auto scores = scorer_.score_models(cid, models);

    // Convert string_view to string for RouteResult (necessary for API contract)
    RouteResult<Scalar> resp{
        .selected_model = scores.empty() ? std::string{} : std::string(scores[0].model_id),
        .alternatives = {},
        .cluster_id = cid,
        .cluster_distance = dist};

    if (scores.size() > 1) {
      resp.alternatives.reserve(scores.size() - 1);
      std::ranges::transform(scores | std::views::drop(1),
                              std::back_inserter(resp.alternatives),
                              [](const ModelScore& s) { return std::string(s.model_id); });
    }
    return resp;
  }

  std::vector<RouteResult<Scalar>> route_batch_impl(
      EmbeddingBatchView<Scalar> view,
      std::optional<std::reference_wrapper<const std::vector<std::string>>> model_filter) {
    if (view.dim != dim_) [[unlikely]] {
      throw std::invalid_argument(std::format("dimension mismatch: {} vs {}", dim_, view.dim));
    }

    auto assignments = backend_->assign_batch(view);
    std::vector<RouteResult<Scalar>> results;
    results.reserve(view.count);
    results.resize(view.count);
    bool has_invalid = false;
    const auto n = std::ssize(results);

    auto models = get_models_to_score(model_filter);

#ifdef _OPENMP
#  pragma omp parallel for schedule(static) reduction(|| : has_invalid)
#endif
    for (ptrdiff_t i = 0; i < n; ++i) {
      const auto& [cid, dist] = assignments[static_cast<size_t>(i)];
      if (cid < 0) {
        has_invalid = true;
        continue;
      }

      auto scores = scorer_.score_models(cid, models);

      std::vector<std::string> alternatives;
      if (scores.size() > 1) {
        alternatives.reserve(scores.size() - 1);
        std::ranges::transform(scores | std::views::drop(1),
                                std::back_inserter(alternatives),
                                [](const ModelScore& s) { return std::string(s.model_id); });
      }

      results[static_cast<size_t>(i)] = RouteResult<Scalar>{
          .selected_model = scores.empty() ? std::string{} : std::string(scores[0].model_id),
          .alternatives = std::move(alternatives),
          .cluster_id = cid,
          .cluster_distance = dist};
    }

    if (has_invalid) [[unlikely]] {
      throw std::runtime_error("no valid cluster");
    }

    return results;
  }

  std::span<const ModelFeatures> get_models_to_score(
      std::optional<std::reference_wrapper<const std::vector<std::string>>> model_filter) {
    if (!model_filter.has_value()) {
      return checkpoint_.models;
    }

    const auto& filter = model_filter->get();
    std::unordered_set<std::string> filter_set(filter.begin(), filter.end());
    auto matching = checkpoint_.models | std::views::filter([&](const ModelFeatures& m) {
                      return filter_set.contains(m.model_id);
                    });
    filtered_models_.assign(matching.begin(), matching.end());
    return std::span<const ModelFeatures>(filtered_models_);
  }

  void init(NordlysCheckpoint checkpoint, Device device = CpuDevice{}) {
    checkpoint_ = std::move(checkpoint);

    const auto& centers = std::get<EmbeddingMatrix<Scalar>>(checkpoint_.cluster_centers);
    dim_ = centers.cols();
    backend_ = create_backend<Scalar>(device);
    backend_->load_centroids(centers.data(), centers.rows(), centers.cols());
  }

  std::unique_ptr<IClusterBackend<Scalar>> backend_;
  ModelScorer scorer_;
  NordlysCheckpoint checkpoint_;
  size_t dim_ = 0;
  std::vector<ModelFeatures> filtered_models_;
};

using Nordlys32 = Nordlys<float>;
using Nordlys64 = Nordlys<double>;
