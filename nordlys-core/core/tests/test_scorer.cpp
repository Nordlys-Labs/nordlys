#include <gtest/gtest.h>

#include <adaptive_core/scorer.hpp>

TEST(ModelScorerTest, EmptyScorer) {
  ModelScorer scorer;
  auto scores = scorer.score_models(0, 0.5f);
  EXPECT_TRUE(scores.empty());
}

TEST(ModelScorerTest, LoadModels) {
  ModelScorer scorer;

  std::vector<ModelFeatures> models;
  ModelFeatures m1;
  m1.model_id = "provider1/model1";
  m1.provider = "provider1";
  m1.model_name = "model1";
  m1.cost_per_1m_input_tokens = 10.0f;
  m1.cost_per_1m_output_tokens = 20.0f;
  m1.error_rates = {0.1f, 0.2f, 0.15f};

  models.push_back(m1);
  scorer.load_models(models);

  auto scores = scorer.score_models(0, 0.5f);
  EXPECT_EQ(scores.size(), 1);
  EXPECT_EQ(scores[0].model_id, "provider1/model1");
}

TEST(ModelScorerTest, CostBiasAffectsScoring) {
  ModelScorer scorer;

  std::vector<ModelFeatures> models;

  // High-accuracy, high-cost model
  ModelFeatures m1;
  m1.model_id = "expensive/accurate";
  m1.provider = "expensive";
  m1.model_name = "accurate";
  m1.cost_per_1m_input_tokens = 100.0f;
  m1.cost_per_1m_output_tokens = 100.0f;
  m1.error_rates = {0.01f};  // Very accurate

  // Low-accuracy, low-cost model
  ModelFeatures m2;
  m2.model_id = "cheap/less_accurate";
  m2.provider = "cheap";
  m2.model_name = "less_accurate";
  m2.cost_per_1m_input_tokens = 1.0f;
  m2.cost_per_1m_output_tokens = 1.0f;
  m2.error_rates = {0.10f};  // Less accurate

  models.push_back(m1);
  models.push_back(m2);
  scorer.load_models(models);

  // With cost_bias = 0 (prefer accuracy), expensive should win
  auto scores_accuracy = scorer.score_models(0, 0.0f);
  EXPECT_EQ(scores_accuracy.size(), 2);
  EXPECT_EQ(scores_accuracy[0].model_id, "expensive/accurate");

  // With cost_bias = 1.0 (prefer cost), cheap should win
  auto scores_cost = scorer.score_models(0, 1.0f);
  EXPECT_EQ(scores_cost.size(), 2);
  EXPECT_EQ(scores_cost[0].model_id, "cheap/less_accurate");
}
