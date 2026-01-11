#include <algorithm>
#include <cmath>
#include <format>
#include <nordlys_core/scorer.hpp>
#include <nordlys_core/tracy.hpp>
#include <ranges>
#include <stdexcept>

std::vector<ModelScore> ModelScorer::score_models(int cluster_id, float cost_bias,
                                                  std::span<const ModelFeatures> models,
                                                  float lambda_min, float lambda_max) const {
  NORDLYS_ZONE;
  if (cluster_id < 0) {
    throw std::invalid_argument(std::format("cluster_id must be non-negative, got {}", cluster_id));
  }

  if (models.empty()) {
    return {};
  }

  auto cost_projection = [](const ModelFeatures& m) { return m.cost_per_1m_tokens(); };
  auto [min_it, max_it] = std::ranges::minmax_element(models, {}, cost_projection);
  float min_cost = cost_projection(*min_it);
  float max_cost = cost_projection(*max_it);
  float cost_range = max_cost - min_cost;

  auto normalize_cost = [=](float cost) {
    if (cost_range <= 0.0f) return 0.0f;
    return (cost - min_cost) / cost_range;
  };

  float lambda = lambda_min + cost_bias * (lambda_max - lambda_min);

  auto create_score = [&](const ModelFeatures& model) -> ModelScore {
    if (cluster_id >= static_cast<int>(model.error_rates.size())) {
      throw std::invalid_argument(
          std::format("cluster_id {} is out of bounds for model '{}' which has {} error rates",
                      cluster_id, model.model_id, model.error_rates.size()));
    }

    float error_rate = model.error_rates[static_cast<std::size_t>(cluster_id)];
    float cost = model.cost_per_1m_tokens();
    float normalized = normalize_cost(cost);
    float score = error_rate + lambda * normalized;

    return ModelScore{.model_id = model.model_id,
                      .score = score,
                      .error_rate = error_rate,
                      .accuracy = 1.0f - error_rate,
                      .cost = cost,
                      .normalized_cost = normalized};
  };

  auto scored = models | std::views::transform(create_score);
  std::vector<ModelScore> scores(scored.begin(), scored.end());
  std::ranges::sort(scores, {}, &ModelScore::score);

  return scores;
}
