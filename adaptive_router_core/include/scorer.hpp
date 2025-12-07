#pragma once
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
  std::string model_id;
  std::string provider;
  std::string model_name;
  std::vector<float> error_rates;  // Per-cluster error rates
  float cost_per_1m_input_tokens;
  float cost_per_1m_output_tokens;

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

  // Load model configurations
  void load_models(const std::vector<ModelFeatures>& models);

  // Set cost normalization range
  void set_cost_range(float min_cost, float max_cost);

  // Set lambda parameter range for cost-accuracy trade-off
  void set_lambda_params(float lambda_min, float lambda_max);

  // Score and rank models for a given cluster
  // cost_bias: 0.0 = prefer accuracy, 1.0 = prefer low cost
  [[nodiscard]] std::vector<ModelScore> score_models(int cluster_id, float cost_bias,
                                                     const std::vector<std::string>& filter = {});

private:
  [[nodiscard]] float normalize_cost(float cost) const noexcept;
  [[nodiscard]] float calculate_lambda(float cost_bias) const noexcept;

  std::vector<ModelFeatures> models_;
  float min_cost_ = 0.0f;
  float max_cost_ = 1.0f;
  float cost_range_ = 1.0f;
  float lambda_min_ = 0.0f;
  float lambda_max_ = 2.0f;
};
