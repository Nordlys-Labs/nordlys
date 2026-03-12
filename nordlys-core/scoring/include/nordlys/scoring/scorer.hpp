#pragma once
#include <span>
#include <string>
#include <string_view>
#include <vector>

struct ModelScore {
  std::string_view model_id;  // References model_id in owned ModelFeatures (no copy)
  float score;                // Per-cluster score (higher is better)
};

struct ModelFeatures {
  std::string model_id;            // e.g., "openai/gpt-4" (single source of truth)
  std::vector<float> scores;       // Per-cluster scores (higher is better)

  // Utility methods (computed, not serialized)
  [[nodiscard]] std::string provider() const {
    auto pos = model_id.find('/');
    return pos != std::string::npos ? model_id.substr(0, pos) : "";
  }

  [[nodiscard]] std::string model_name() const {
    auto pos = model_id.find('/');
    return pos != std::string::npos ? model_id.substr(pos + 1) : model_id;
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

  // Score and rank models for a given cluster by score (higher is better)
  [[nodiscard]] std::vector<ModelScore> score_models(int cluster_id,
                                                     std::span<const ModelFeatures> models) const;
};
