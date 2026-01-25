#include <algorithm>
#include <cmath>
#include <format>
#include <nordlys_core/scorer.hpp>
#include <ranges>
#include <stdexcept>

std::vector<ModelScore> ModelScorer::score_models(int cluster_id, float cost_bias,
                                                  std::span<const ModelFeatures> models,
                                                  float lambda_min, float lambda_max) const {
  if (cluster_id < 0) {
    throw std::invalid_argument(std::format("cluster_id must be non-negative, got {}", cluster_id));
  }

  if (models.empty()) {
    return {};
  }

  // Single-pass: find min/max cost and compute scores simultaneously
  float min_cost = std::numeric_limits<float>::max();
  float max_cost = std::numeric_limits<float>::lowest();
  std::vector<ModelScore> scores;
  scores.reserve(models.size());

  float lambda = lambda_min + cost_bias * (lambda_max - lambda_min);

  // First pass: compute scores and track min/max cost
  for (const auto& model : models) {
    if (cluster_id >= static_cast<int>(model.error_rates.size())) {
      throw std::invalid_argument(
          std::format("cluster_id {} is out of bounds for model '{}' which has {} error rates",
                      cluster_id, model.model_id, model.error_rates.size()));
    }

    float error_rate = model.error_rates[static_cast<std::size_t>(cluster_id)];
    float cost = model.cost_per_1m_tokens();
    
    // Track min/max cost
    if (cost < min_cost) min_cost = cost;
    if (cost > max_cost) max_cost = cost;

    // Store score with unnormalized cost for now
    // Use string_view to avoid string copy - ModelFeatures are owned by Nordlys
    // and will outlive the scores vector. The string_view references the model_id
    // in the owned ModelFeatures, avoiding a copy until RouteResult is created.
    scores.emplace_back(ModelScore{
        .model_id = model.model_id,  // string_view from owned ModelFeatures (no copy)
        .score = 0.0f,               // score - will compute after normalization
        .error_rate = error_rate,
        .accuracy = 1.0f - error_rate,
        .cost = cost,
        .normalized_cost = 0.0f});   // normalized_cost - will compute after normalization
  }

  // Compute cost range and normalize
  float cost_range = max_cost - min_cost;
  auto normalize_cost = [=](float cost) {
    if (cost_range <= 0.0f) return 0.0f;
    return (cost - min_cost) / cost_range;
  };

  // Second pass: normalize costs and compute final scores
  for (auto& score : scores) {
    score.normalized_cost = normalize_cost(score.cost);
    score.score = score.error_rate + lambda * score.normalized_cost;
  }

  std::ranges::sort(scores, {}, &ModelScore::score);

  return scores;
}
