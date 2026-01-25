#pragma once
#include <span>
#include <string>
#include <string_view>
#include <vector>

struct ModelScore {
  std::string_view model_id;  // References model_id in owned ModelFeatures (no copy)
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

  // Score and rank models for a given cluster by error rate
  [[nodiscard]] std::vector<ModelScore> score_models(int cluster_id,
                                                     std::span<const ModelFeatures> models) const;
};
