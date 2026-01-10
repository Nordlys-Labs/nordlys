#include <algorithm>
#include <cmath>
#include <format>
#include <nordlys_core/scorer.hpp>
#include <nordlys_core/tracy.hpp>
#include <ranges>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

float ModelScorer::normalize_cost(float cost) const noexcept {
  if (cost_range_ <= 0.0f) return 0.0f;
  return (cost - min_cost_) / cost_range_;
}

float ModelScorer::calculate_lambda(float cost_bias) const noexcept {
  // cost_bias: 0.0 = prefer accuracy, 1.0 = prefer low cost
  return lambda_min_ + cost_bias * (lambda_max_ - lambda_min_);
}

void ModelScorer::load_models(const std::vector<ModelFeatures>& models) {
  models_ = models;

  // Calculate cost range using ranges::minmax_element with projection
  if (!models.empty()) {
    auto cost_projection = [](const auto& model) { return model.cost_per_1m_tokens(); };
    auto [min_it, max_it] = std::ranges::minmax_element(models, {}, cost_projection);

    min_cost_ = cost_projection(*min_it);
    max_cost_ = cost_projection(*max_it);
    cost_range_ = max_cost_ - min_cost_;
  }
}

void ModelScorer::set_cost_range(float min_cost, float max_cost) {
  min_cost_ = min_cost;
  max_cost_ = max_cost;
  cost_range_ = max_cost - min_cost;
}

void ModelScorer::set_lambda_params(float lambda_min, float lambda_max) {
  lambda_min_ = lambda_min;
  lambda_max_ = lambda_max;
}

std::vector<ModelScore> ModelScorer::score_models(int cluster_id, float cost_bias,
                                                  const std::vector<std::string>& filter) {
  NORDLYS_ZONE;
  // Validate cluster_id
  if (cluster_id < 0) {
    throw std::invalid_argument(std::format("cluster_id must be non-negative, got {}", cluster_id));
  }

  // Build filter set if provided
  std::unordered_set<std::string> filter_set(filter.begin(), filter.end());
  bool use_filter = !filter.empty();

  float lambda = calculate_lambda(cost_bias);

  // Create scoring function
  auto create_score = [&](const ModelFeatures& model) -> ModelScore {
    // Validate cluster_id is within bounds for this model's error_rates
    if (cluster_id >= static_cast<int>(model.error_rates.size())) {
      throw std::invalid_argument(
          std::format("cluster_id {} is out of bounds for model '{}' which has {} error rates",
                      cluster_id, model.model_id, model.error_rates.size()));
    }

    float error_rate = model.error_rates[static_cast<std::size_t>(cluster_id)];
    float cost = model.cost_per_1m_tokens();
    float normalized_cost = normalize_cost(cost);

    // Score = error_rate + lambda * normalized_cost (lower is better)
    float score = error_rate + lambda * normalized_cost;

    return ModelScore{.model_id = model.model_id,
                      .score = score,
                      .error_rate = error_rate,
                      .accuracy = 1.0f - error_rate,
                      .cost = cost,
                      .normalized_cost = normalized_cost};
  };

  // Filter models and transform to scores using ranges pipeline
  auto filtered_models = models_ | std::views::filter([&](const auto& model) {
                           return !use_filter || filter_set.contains(model.model_id);
                         })
                         | std::views::transform(create_score);

  // Convert to vector and sort
  std::vector<ModelScore> scores(filtered_models.begin(), filtered_models.end());
  std::ranges::sort(scores, {}, &ModelScore::score);

  return scores;
}
