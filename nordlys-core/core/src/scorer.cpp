#include <algorithm>
#include <cmath>
#include <format>
#include <nordlys_core/scorer.hpp>
#include <ranges>
#include <stdexcept>

std::vector<ModelScore> ModelScorer::score_models(int cluster_id,
                                                  std::span<const ModelFeatures> models) const {
  if (cluster_id < 0) {
    throw std::invalid_argument(std::format("cluster_id must be non-negative, got {}", cluster_id));
  }

  if (models.empty()) {
    return {};
  }

  std::vector<ModelScore> scores;
  scores.reserve(models.size());

  // Compute scores based on error rate only
  for (const auto& model : models) {
    if (cluster_id >= static_cast<int>(model.error_rates.size())) {
      throw std::invalid_argument(
          std::format("cluster_id {} is out of bounds for model '{}' which has {} error rates",
                      cluster_id, model.model_id, model.error_rates.size()));
    }

    float error_rate = model.error_rates[static_cast<std::size_t>(cluster_id)];
    float cost = model.cost_per_1m_tokens();

    // Use string_view to avoid string copy - ModelFeatures are owned by Nordlys
    // and will outlive the scores vector. The string_view references the model_id
    // in the owned ModelFeatures, avoiding a copy until RouteResult is created.
    scores.emplace_back(
        ModelScore{.model_id = model.model_id,  // string_view from owned ModelFeatures (no copy)
                   .score = error_rate,         // score = error_rate (lower is better)
                   .error_rate = error_rate,
                   .accuracy = 1.0f - error_rate,
                   .cost = cost,
                   .normalized_cost = 0.0f});  // normalized_cost not used anymore
  }

  // Sort by score (error_rate) - lower is better
  std::ranges::sort(scores, {}, &ModelScore::score);

  return scores;
}
