#pragma once
#include <span>
#include <string>
#include <vector>

struct ModelScore {
  std::string model_id;
  float score;
  float error_rate;
  float accuracy;
  float cost;
  float normalized_cost;
};

struct ModelFeatures {
  std::string model_id;            // e.g., "openai/gpt-4" (single source of truth)
  std::vector<float> error_rates;  // Per-cluster error rates
  float cost_per_1m_input_tokens;
  float cost_per_1m_output_tokens;

  // Utility methods (computed, not serialized)
  [[nodiscard]] std::string provider() const {
    auto pos = model_id.find('/');
    return pos != std::string::npos ? model_id.substr(0, pos) : "";
  }

  [[nodiscard]] std::string model_name() const {
    auto pos = model_id.find('/');
    return pos != std::string::npos ? model_id.substr(pos + 1) : model_id;
  }

  // Average cost per 1M tokens
  [[nodiscard]] constexpr float cost_per_1m_tokens() const noexcept {
    return (cost_per_1m_input_tokens + cost_per_1m_output_tokens) / 2.0f;
  }
};

class ModelScorer {
public:
  ModelScorer() = default;
  ~ModelScorer() = default;

  // Movable
  ModelScorer(ModelScorer&&) = default;
  ModelScorer& operator=(ModelScorer&&) = default;
  ModelScorer(const ModelScorer&) = delete;
  ModelScorer& operator=(const ModelScorer&) = delete;

  // Score and rank models for a given cluster
  // cost_bias: 0.0 = prefer accuracy, 1.0 = prefer low cost
  // lambda_min: minimum lambda for cost-accuracy trade-off (default 0.0)
  // lambda_max: maximum lambda for cost-accuracy trade-off (default 2.0)
  [[nodiscard]] std::vector<ModelScore> score_models(int cluster_id, float cost_bias,
                                                     std::span<const ModelFeatures> models,
                                                     float lambda_min = 0.0f,
                                                     float lambda_max = 2.0f) const;
};
