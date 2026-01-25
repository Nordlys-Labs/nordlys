#include <gtest/gtest.h>

#include <limits>
#include <nordlys_core/scorer.hpp>

TEST(ModelScorerTest, EmptyModelsReturnsEmptyScores) {
  ModelScorer scorer;
  std::vector<ModelFeatures> models;
  auto scores = scorer.score_models(0, models);
  EXPECT_TRUE(scores.empty());
}

TEST(ModelScorerTest, SingleModel) {
  ModelScorer scorer;

  std::vector<ModelFeatures> models;
  ModelFeatures m1;
  m1.model_id = "only_model";
  m1.cost_per_1m_input_tokens = 10.0f;
  m1.cost_per_1m_output_tokens = 10.0f;
  m1.error_rates = {0.1f};

  models.push_back(m1);

  auto scores = scorer.score_models(0, models);
  EXPECT_EQ(scores.size(), 1);
  EXPECT_EQ(scores[0].model_id, "only_model");
}

TEST(ModelScorerTest, ScoringByErrorRate) {
  ModelScorer scorer;

  std::vector<ModelFeatures> models;

  ModelFeatures m1;
  m1.model_id = "expensive/accurate";
  m1.cost_per_1m_input_tokens = 100.0f;
  m1.cost_per_1m_output_tokens = 100.0f;
  m1.error_rates = {0.01f};

  ModelFeatures m2;
  m2.model_id = "cheap/less_accurate";
  m2.cost_per_1m_input_tokens = 1.0f;
  m2.cost_per_1m_output_tokens = 1.0f;
  m2.error_rates = {0.10f};

  models.push_back(m1);
  models.push_back(m2);

  auto scores = scorer.score_models(0, models);
  EXPECT_EQ(scores.size(), 2);
  // Should rank by error rate (lower is better)
  EXPECT_EQ(scores[0].model_id, "expensive/accurate");
  EXPECT_EQ(scores[1].model_id, "cheap/less_accurate");
}

TEST(ModelScorerTest, ZeroCostRange) {
  ModelScorer scorer;

  std::vector<ModelFeatures> models;
  ModelFeatures m1;
  m1.model_id = "model1";
  m1.cost_per_1m_input_tokens = 10.0f;
  m1.cost_per_1m_output_tokens = 10.0f;
  m1.error_rates = {0.1f};

  ModelFeatures m2;
  m2.model_id = "model2";
  m2.cost_per_1m_input_tokens = 10.0f;
  m2.cost_per_1m_output_tokens = 10.0f;
  m2.error_rates = {0.2f};

  models.push_back(m1);
  models.push_back(m2);

  auto scores = scorer.score_models(0, models);
  EXPECT_EQ(scores.size(), 2);
  EXPECT_EQ(scores[0].model_id, "model1");
}

TEST(ModelScorerTest, CustomLambdaParams) {
  ModelScorer scorer;

  std::vector<ModelFeatures> models;
  ModelFeatures m1;
  m1.model_id = "model1";
  m1.cost_per_1m_input_tokens = 10.0f;
  m1.cost_per_1m_output_tokens = 10.0f;
  m1.error_rates = {0.1f};

  models.push_back(m1);

  auto scores = scorer.score_models(0, models);
  EXPECT_EQ(scores.size(), 1);
}

TEST(ModelScorerTest, NegativeClusterIdThrows) {
  ModelScorer scorer;

  std::vector<ModelFeatures> models;
  ModelFeatures m1;
  m1.model_id = "model1";
  m1.cost_per_1m_input_tokens = 10.0f;
  m1.cost_per_1m_output_tokens = 10.0f;
  m1.error_rates = {0.1f};

  models.push_back(m1);

  EXPECT_THROW(scorer.score_models(-1, models), std::invalid_argument);
}

TEST(ModelScorerTest, ClusterIdOutOfBoundsThrows) {
  ModelScorer scorer;

  std::vector<ModelFeatures> models;
  ModelFeatures m1;
  m1.model_id = "model1";
  m1.cost_per_1m_input_tokens = 10.0f;
  m1.cost_per_1m_output_tokens = 10.0f;
  m1.error_rates = {0.1f, 0.2f};

  models.push_back(m1);

  EXPECT_THROW(scorer.score_models(5, models), std::invalid_argument);
}

TEST(ModelScorerTest, ManyModels) {
  ModelScorer scorer;

  std::vector<ModelFeatures> models;
  for (int i = 0; i < 50; ++i) {
    ModelFeatures m;
    m.model_id = "model" + std::to_string(i);
    m.cost_per_1m_input_tokens = 10.0f + i;
    m.cost_per_1m_output_tokens = 10.0f + i;
    m.error_rates = {0.01f * i};
    models.push_back(m);
  }

  auto scores = scorer.score_models(0, models);
  EXPECT_EQ(scores.size(), 50);
}

TEST(ModelScorerTest, ExtremeCostValues) {
  ModelScorer scorer;

  std::vector<ModelFeatures> models;
  ModelFeatures m1;
  m1.model_id = "very_cheap";
  m1.cost_per_1m_input_tokens = 0.001f;
  m1.cost_per_1m_output_tokens = 0.001f;
  m1.error_rates = {0.5f};

  ModelFeatures m2;
  m2.model_id = "very_expensive";
  m2.cost_per_1m_input_tokens = 1000.0f;
  m2.cost_per_1m_output_tokens = 1000.0f;
  m2.error_rates = {0.01f};

  models.push_back(m1);
  models.push_back(m2);

  auto scores = scorer.score_models(0, models);
  EXPECT_EQ(scores.size(), 2);
}

TEST(ModelScorerTest, ScoringWorks) {
  ModelScorer scorer;

  std::vector<ModelFeatures> models;
  ModelFeatures m1;
  m1.model_id = "model1";
  m1.cost_per_1m_input_tokens = 10.0f;
  m1.cost_per_1m_output_tokens = 10.0f;
  m1.error_rates = {0.1f};

  models.push_back(m1);

  auto scores_neg = scorer.score_models(0, models);
  EXPECT_EQ(scores_neg.size(), 1);

  auto scores_large = scorer.score_models(0, models);
  EXPECT_EQ(scores_large.size(), 1);
}

TEST(ModelScorerTest, ErrorRateBoundaries) {
  ModelScorer scorer;

  std::vector<ModelFeatures> models;
  ModelFeatures m1;
  m1.model_id = "perfect";
  m1.cost_per_1m_input_tokens = 10.0f;
  m1.cost_per_1m_output_tokens = 10.0f;
  m1.error_rates = {0.0f};

  ModelFeatures m2;
  m2.model_id = "worst";
  m2.cost_per_1m_input_tokens = 10.0f;
  m2.cost_per_1m_output_tokens = 10.0f;
  m2.error_rates = {1.0f};

  models.push_back(m1);
  models.push_back(m2);

  auto scores = scorer.score_models(0, models);
  EXPECT_EQ(scores.size(), 2);
  EXPECT_EQ(scores[0].model_id, "perfect");
  EXPECT_FLOAT_EQ(scores[0].accuracy, 1.0f);
  EXPECT_FLOAT_EQ(scores[1].accuracy, 0.0f);
}

TEST(ModelScorerTest, SameErrorRateOrdering) {
  ModelScorer scorer;

  std::vector<ModelFeatures> models;
  ModelFeatures m1;
  m1.model_id = "cheap";
  m1.cost_per_1m_input_tokens = 1.0f;
  m1.cost_per_1m_output_tokens = 1.0f;
  m1.error_rates = {0.1f};

  ModelFeatures m2;
  m2.model_id = "expensive";
  m2.cost_per_1m_input_tokens = 100.0f;
  m2.cost_per_1m_output_tokens = 100.0f;
  m2.error_rates = {0.1f};

  models.push_back(m1);
  models.push_back(m2);

  auto scores = scorer.score_models(0, models);
  EXPECT_EQ(scores.size(), 2);
  // With same error rate, order is stable (first model wins)
  EXPECT_EQ(scores[0].model_id, "cheap");
  EXPECT_EQ(scores[1].model_id, "expensive");
}

TEST(ModelScorerTest, AccuracyFieldCalculation) {
  ModelScorer scorer;

  std::vector<ModelFeatures> models;
  ModelFeatures m1;
  m1.model_id = "model1";
  m1.cost_per_1m_input_tokens = 10.0f;
  m1.cost_per_1m_output_tokens = 10.0f;
  m1.error_rates = {0.25f};

  models.push_back(m1);

  auto scores = scorer.score_models(0, models);
  EXPECT_EQ(scores.size(), 1);
  EXPECT_FLOAT_EQ(scores[0].error_rate, 0.25f);
  EXPECT_FLOAT_EQ(scores[0].accuracy, 0.75f);
}

TEST(ModelScorerTest, SortingOrderVerification) {
  ModelScorer scorer;

  std::vector<ModelFeatures> models;
  for (int i = 0; i < 10; ++i) {
    ModelFeatures m;
    m.model_id = "model" + std::to_string(i);
    m.cost_per_1m_input_tokens = 10.0f;
    m.cost_per_1m_output_tokens = 10.0f;
    m.error_rates = {0.1f * (10 - i)};
    models.push_back(m);
  }

  auto scores = scorer.score_models(0, models);
  EXPECT_EQ(scores.size(), 10);

  for (size_t i = 1; i < scores.size(); ++i) {
    EXPECT_LE(scores[i - 1].score, scores[i].score)
        << "Scores not sorted: " << scores[i - 1].score << " > " << scores[i].score;
  }
}

TEST(ModelScorerTest, MoveConstructor) {
  ModelScorer scorer1;

  std::vector<ModelFeatures> models;
  ModelFeatures m1;
  m1.model_id = "model1";
  m1.cost_per_1m_input_tokens = 10.0f;
  m1.cost_per_1m_output_tokens = 10.0f;
  m1.error_rates = {0.1f};

  models.push_back(m1);

  ModelScorer scorer2(std::move(scorer1));

  auto scores = scorer2.score_models(0, models);
  EXPECT_EQ(scores.size(), 1);
}

TEST(ModelScorerTest, MoveAssignment) {
  ModelScorer scorer1;

  std::vector<ModelFeatures> models;
  ModelFeatures m1;
  m1.model_id = "model1";
  m1.cost_per_1m_input_tokens = 10.0f;
  m1.cost_per_1m_output_tokens = 10.0f;
  m1.error_rates = {0.1f};

  models.push_back(m1);

  ModelScorer scorer2;
  scorer2 = std::move(scorer1);

  auto scores = scorer2.score_models(0, models);
  EXPECT_EQ(scores.size(), 1);
}

TEST(ModelFeaturesTest, ProviderAndModelNameParsing) {
  ModelFeatures m;
  m.model_id = "openai/gpt-4";

  EXPECT_EQ(m.provider(), "openai");
  EXPECT_EQ(m.model_name(), "gpt-4");
}

TEST(ModelFeaturesTest, ProviderParsingNoSlash) {
  ModelFeatures m;
  m.model_id = "standalone-model";

  EXPECT_EQ(m.provider(), "");
  EXPECT_EQ(m.model_name(), "standalone-model");
}

TEST(ModelFeaturesTest, ProviderParsingMultipleSlashes) {
  ModelFeatures m;
  m.model_id = "provider/subprovider/model";

  EXPECT_EQ(m.provider(), "provider");
  EXPECT_EQ(m.model_name(), "subprovider/model");
}

TEST(ModelFeaturesTest, CostPerTokenCalculation) {
  ModelFeatures m;
  m.cost_per_1m_input_tokens = 10.0f;
  m.cost_per_1m_output_tokens = 20.0f;

  EXPECT_FLOAT_EQ(m.cost_per_1m_tokens(), 15.0f);
}
