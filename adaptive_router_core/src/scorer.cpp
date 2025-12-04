#include "scorer.hpp"
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

float ModelScorer::normalize_cost(float cost) const {
    if (cost_range_ <= 0.0f) return 0.0f;
    return (cost - min_cost_) / cost_range_;
}

float ModelScorer::calculate_lambda(float cost_bias) const {
    // cost_bias: 0.0 = prefer accuracy, 1.0 = prefer low cost
    return lambda_min_ + cost_bias * (lambda_max_ - lambda_min_);
}

void ModelScorer::load_models(const std::vector<ModelFeatures>& models) {
    models_ = models;

    // Calculate cost range
    if (!models.empty()) {
        min_cost_ = std::numeric_limits<float>::max();
        max_cost_ = std::numeric_limits<float>::lowest();

        for (const auto& m : models) {
            float cost = m.cost_per_1m_tokens();
            min_cost_ = std::min(min_cost_, cost);
            max_cost_ = std::max(max_cost_, cost);
        }
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

std::vector<ModelScore> ModelScorer::score_models(
    int cluster_id,
    float cost_bias,
    const std::vector<std::string>& filter
) {
    std::vector<ModelScore> scores;

    // Build filter set if provided
    std::unordered_set<std::string> filter_set(filter.begin(), filter.end());
    bool use_filter = !filter.empty();

    float lambda = calculate_lambda(cost_bias);

    for (const auto& model : models_) {
        // Apply filter if provided
        if (use_filter && filter_set.find(model.model_id) == filter_set.end()) {
            continue;
        }

        // Get error rate for this cluster
        float error_rate = 0.0f;
        if (cluster_id >= 0 && cluster_id < static_cast<int>(model.error_rates.size())) {
            error_rate = model.error_rates[cluster_id];
        }

        float cost = model.cost_per_1m_tokens();
        float normalized_cost = normalize_cost(cost);

        // Score = error_rate + lambda * normalized_cost
        // Lower score is better
        float score = error_rate + lambda * normalized_cost;

        scores.push_back(ModelScore{
            .model_id = model.model_id,
            .score = score,
            .error_rate = error_rate,
            .accuracy = 1.0f - error_rate,
            .cost = cost,
            .normalized_cost = normalized_cost
        });
    }

    // Sort by score (ascending - lower is better)
    std::sort(scores.begin(), scores.end(),
        [](const ModelScore& a, const ModelScore& b) {
            return a.score < b.score;
        });

    return scores;
}
