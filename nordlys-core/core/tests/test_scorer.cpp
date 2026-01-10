#include <gtest/gtest.h>

#include <nordlys_core/scorer.hpp>
#include <limits>

// =============================================================================
// Basic Functionality Tests
// =============================================================================

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
  scorer.load_models(models);

  auto scores_accuracy = scorer.score_models(0, 0.0f);
  EXPECT_EQ(scores_accuracy.size(), 2);
  EXPECT_EQ(scores_accuracy[0].model_id, "expensive/accurate");

  auto scores_cost = scorer.score_models(0, 1.0f);
  EXPECT_EQ(scores_cost.size(), 2);
  EXPECT_EQ(scores_cost[0].model_id, "cheap/less_accurate");
}

// =============================================================================
// ModelFeatures Tests
// =============================================================================

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

// =============================================================================
// Edge Cases Tests
// =============================================================================

TEST(ModelScorerTest, SetCostRangeCustom) {
  ModelScorer scorer;

  std::vector<ModelFeatures> models;
  ModelFeatures m1;
  m1.model_id = "model1";
  m1.cost_per_1m_input_tokens = 10.0f;
  m1.cost_per_1m_output_tokens = 10.0f;
  m1.error_rates = {0.1f};

  models.push_back(m1);
  scorer.load_models(models);

  scorer.set_cost_range(5.0f, 25.0f);

  auto scores = scorer.score_models(0, 0.5f);
  EXPECT_EQ(scores.size(), 1);
  EXPECT_GE(scores[0].normalized_cost, 0.0f);
  EXPECT_LE(scores[0].normalized_cost, 1.0f);
}

TEST(ModelScorerTest, SetLambdaParamsCustom) {
  ModelScorer scorer;

  std::vector<ModelFeatures> models;
  ModelFeatures m1;
  m1.model_id = "model1";
  m1.cost_per_1m_input_tokens = 10.0f;
  m1.cost_per_1m_output_tokens = 10.0f;
  m1.error_rates = {0.1f};

  models.push_back(m1);
  scorer.load_models(models);

  scorer.set_lambda_params(0.5f, 3.0f);

  auto scores = scorer.score_models(0, 0.5f);
  EXPECT_EQ(scores.size(), 1);
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
  scorer.load_models(models);

  auto scores = scorer.score_models(0, 0.5f);
  EXPECT_EQ(scores.size(), 2);
  EXPECT_EQ(scores[0].model_id, "model1");
}

TEST(ModelScorerTest, FilterWithNonexistentModels) {
  ModelScorer scorer;

  std::vector<ModelFeatures> models;
  ModelFeatures m1;
  m1.model_id = "existing/model";
  m1.cost_per_1m_input_tokens = 10.0f;
  m1.cost_per_1m_output_tokens = 10.0f;
  m1.error_rates = {0.1f};

  models.push_back(m1);
  scorer.load_models(models);

  std::vector<std::string> filter = {"nonexistent/model"};
  auto scores = scorer.score_models(0, 0.5f, filter);
  EXPECT_TRUE(scores.empty());
}

TEST(ModelScorerTest, EmptyFilterUsesAllModels) {
  ModelScorer scorer;

  std::vector<ModelFeatures> models;
  ModelFeatures m1;
  m1.model_id = "model1";
  m1.cost_per_1m_input_tokens = 10.0f;
  m1.cost_per_1m_output_tokens = 10.0f;
  m1.error_rates = {0.1f};

  ModelFeatures m2;
  m2.model_id = "model2";
  m2.cost_per_1m_input_tokens = 15.0f;
  m2.cost_per_1m_output_tokens = 15.0f;
  m2.error_rates = {0.2f};

  models.push_back(m1);
  models.push_back(m2);
  scorer.load_models(models);

  std::vector<std::string> empty_filter;
  auto scores = scorer.score_models(0, 0.5f, empty_filter);
  EXPECT_EQ(scores.size(), 2);
}

TEST(ModelScorerTest, FilterExcludesAllModels) {
  ModelScorer scorer;

  std::vector<ModelFeatures> models;
  ModelFeatures m1;
  m1.model_id = "model1";
  m1.cost_per_1m_input_tokens = 10.0f;
  m1.cost_per_1m_output_tokens = 10.0f;
  m1.error_rates = {0.1f};

  models.push_back(m1);
  scorer.load_models(models);

  std::vector<std::string> filter = {"different_model"};
  auto scores = scorer.score_models(0, 0.5f, filter);
  EXPECT_TRUE(scores.empty());
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
  scorer.load_models(models);

  EXPECT_THROW(scorer.score_models(-1, 0.5f), std::invalid_argument);
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
  scorer.load_models(models);

  EXPECT_THROW(scorer.score_models(5, 0.5f), std::invalid_argument);
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
  scorer.load_models(models);

  auto scores = scorer.score_models(0, 0.5f);
  EXPECT_EQ(scores.size(), 1);
  EXPECT_EQ(scores[0].model_id, "only_model");
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

  scorer.load_models(models);

  auto scores = scorer.score_models(0, 0.5f);
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
  scorer.load_models(models);

  auto scores = scorer.score_models(0, 0.5f);
  EXPECT_EQ(scores.size(), 2);
}

TEST(ModelScorerTest, ExtremeCostBias) {
  ModelScorer scorer;

  std::vector<ModelFeatures> models;
  ModelFeatures m1;
  m1.model_id = "model1";
  m1.cost_per_1m_input_tokens = 10.0f;
  m1.cost_per_1m_output_tokens = 10.0f;
  m1.error_rates = {0.1f};

  models.push_back(m1);
  scorer.load_models(models);

  auto scores_neg = scorer.score_models(0, -1.0f);
  EXPECT_EQ(scores_neg.size(), 1);

  auto scores_large = scorer.score_models(0, 10.0f);
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
  scorer.load_models(models);

  auto scores = scorer.score_models(0, 0.5f);
  EXPECT_EQ(scores.size(), 2);
  EXPECT_EQ(scores[0].model_id, "perfect");
  EXPECT_FLOAT_EQ(scores[0].accuracy, 1.0f);
  EXPECT_FLOAT_EQ(scores[1].accuracy, 0.0f);
}

TEST(ModelScorerTest, NormalizedCostCalculation) {
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
  scorer.load_models(models);

  auto scores = scorer.score_models(0, 0.0f);
  EXPECT_EQ(scores.size(), 2);

  EXPECT_FLOAT_EQ(scores[0].normalized_cost, 0.0f);
  EXPECT_FLOAT_EQ(scores[1].normalized_cost, 1.0f);
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
  scorer.load_models(models);

  auto scores = scorer.score_models(0, 0.5f);
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

  scorer.load_models(models);

  auto scores = scorer.score_models(0, 0.0f);
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
  scorer1.load_models(models);

  ModelScorer scorer2(std::move(scorer1));

  auto scores = scorer2.score_models(0, 0.5f);
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
  scorer1.load_models(models);

  ModelScorer scorer2;
  scorer2 = std::move(scorer1);

  auto scores = scorer2.score_models(0, 0.5f);
  EXPECT_EQ(scores.size(), 1);
}

TEST(ModelScorerTest, FilterMatchesSubset) {
  ModelScorer scorer;

  std::vector<ModelFeatures> models;
  for (int i = 0; i < 5; ++i) {
    ModelFeatures m;
    m.model_id = "model" + std::to_string(i);
    m.cost_per_1m_input_tokens = 10.0f + i;
    m.cost_per_1m_output_tokens = 10.0f + i;
    m.error_rates = {0.1f};
    models.push_back(m);
  }

  scorer.load_models(models);

  std::vector<std::string> filter = {"model1", "model3"};
  auto scores = scorer.score_models(0, 0.5f, filter);
  EXPECT_EQ(scores.size(), 2);
  EXPECT_TRUE(scores[0].model_id == "model1" || scores[0].model_id == "model3");
  EXPECT_TRUE(scores[1].model_id == "model1" || scores[1].model_id == "model3");
}
