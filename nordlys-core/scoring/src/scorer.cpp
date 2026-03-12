#include <algorithm>
#include <cmath>
#include <format>
#include <nordlys/scoring/scorer.hpp>
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

  // Compute scores - higher is better
  for (const auto& model : models) {
    if (cluster_id >= static_cast<int>(model.scores.size())) {
      throw std::invalid_argument(
          std::format("cluster_id {} is out of bounds for model '{}' which has {} scores",
                      cluster_id, model.model_id, model.scores.size()));
    }

    float score = model.scores[static_cast<std::size_t>(cluster_id)];

    // Use string_view to avoid string copy - ModelFeatures are owned by Nordlys
    // and will outlive the scores vector. The string_view references the model_id
    // in the owned ModelFeatures, avoiding a copy until RouteResult is created.
    scores.emplace_back(
        ModelScore{.model_id = model.model_id,  // string_view from owned ModelFeatures (no copy)
                   .score = score});            // higher is better
  }

  // Sort by score descending - higher is better
  std::ranges::sort(scores, std::greater<float>{}, &ModelScore::score);

  return scores;
}
