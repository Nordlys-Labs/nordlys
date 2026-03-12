#include <gtest/gtest.h>

#include <limits>
#include <nordlys/scoring/scorer.hpp>

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
  m1.scores = {0.1f};

  models.push_back(m1);

  auto scores = scorer.score_models(0, models);
  EXPECT_EQ(scores.size(), 1);
  EXPECT_EQ(scores[0].model_id, "only_model");
}

TEST(ModelScorerTest, ScoringByScore) {
  ModelScorer scorer;

  std::vector<ModelFeatures> models;

  ModelFeatures m1;
  m1.model_id = "expensive/accurate";
  m1.scores = {0.99f};

  ModelFeatures m2;
  m2.model_id = "cheap/less_accurate";
  m2.scores = {0.90f};

  models.push_back(m1);
  models.push_back(m2);

  auto scores = scorer.score_models(0, models);
  EXPECT_EQ(scores.size(), 2);
  // Should rank by score (higher is better)
  EXPECT_EQ(scores[0].model_id, "expensive/accurate");
  EXPECT_EQ(scores[1].model_id, "cheap/less_accurate");
}

TEST(ModelScorerTest, ScoreSorting) {
  ModelScorer scorer;

  std::vector<ModelFeatures> models;
  ModelFeatures m1;
  m1.model_id = "model1";
  m1.scores = {0.1f};

  ModelFeatures m2;
  m2.model_id = "model2";
  m2.scores = {0.2f};

  models.push_back(m1);
  models.push_back(m2);

  auto scores = scorer.score_models(0, models);
  EXPECT_EQ(scores.size(), 2);
  // Higher score should be first
  EXPECT_EQ(scores[0].model_id, "model2");
}

TEST(ModelScorerTest, CustomLambdaParams) {
  ModelScorer scorer;

  std::vector<ModelFeatures> models;
  ModelFeatures m1;
  m1.model_id = "model1";

  m1.scores = {0.1f};

  models.push_back(m1);

  auto scores = scorer.score_models(0, models);
  EXPECT_EQ(scores.size(), 1);
}

TEST(ModelScorerTest, NegativeClusterIdThrows) {
  ModelScorer scorer;

  std::vector<ModelFeatures> models;
  ModelFeatures m1;
  m1.model_id = "model1";

  m1.scores = {0.1f};

  models.push_back(m1);

  EXPECT_THROW(scorer.score_models(-1, models), std::invalid_argument);
}

TEST(ModelScorerTest, ClusterIdOutOfBoundsThrows) {
  ModelScorer scorer;

  std::vector<ModelFeatures> models;
  ModelFeatures m1;
  m1.model_id = "model1";

  m1.scores = {0.1f, 0.2f};

  models.push_back(m1);

  EXPECT_THROW(scorer.score_models(5, models), std::invalid_argument);
}

TEST(ModelScorerTest, ManyModels) {
  ModelScorer scorer;

  std::vector<ModelFeatures> models;
  for (int i = 0; i < 50; ++i) {
    ModelFeatures m;
    m.model_id = "model" + std::to_string(i);

    m.scores = {0.01f * i};
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
  m1.scores = {0.5f};

  ModelFeatures m2;
  m2.model_id = "model2";
  m2.scores = {0.01f};

  models.push_back(m1);
  models.push_back(m2);

  auto scores = scorer.score_models(0, models);
  EXPECT_EQ(scores.size(), 2);
}

TEST(ModelScorerTest, ErrorRateBoundaries) {
  ModelScorer scorer;

  std::vector<ModelFeatures> models;
  ModelFeatures m1;
  m1.model_id = "perfect";
  m1.scores = {1.0f};

  ModelFeatures m2;
  m2.model_id = "worst";
  m2.scores = {0.0f};

  models.push_back(m1);
  models.push_back(m2);

  auto scores = scorer.score_models(0, models);
  EXPECT_EQ(scores.size(), 2);
  EXPECT_EQ(scores[0].model_id, "perfect");
  EXPECT_EQ(scores[1].model_id, "worst");
}

TEST(ModelScorerTest, SameScoreOrdering) {
  ModelScorer scorer;

  std::vector<ModelFeatures> models;
  ModelFeatures m1;
  m1.model_id = "model1";
  m1.scores = {0.5f};

  ModelFeatures m2;
  m2.model_id = "model2";
  m2.scores = {0.5f};

  models.push_back(m1);
  models.push_back(m2);

  auto scores = scorer.score_models(0, models);
  EXPECT_EQ(scores.size(), 2);
  EXPECT_EQ(scores[0].score, 0.5f);
}

TEST(ModelScorerTest, AccuracyFieldCalculation) {
  ModelScorer scorer;

  std::vector<ModelFeatures> models;
  ModelFeatures m1;
  m1.model_id = "model1";

  m1.scores = {0.25f};

  models.push_back(m1);

  auto scores = scorer.score_models(0, models);
  EXPECT_EQ(scores.size(), 1);
  EXPECT_FLOAT_EQ(scores[0].score, 0.25f);
}

TEST(ModelScorerTest, SortingOrderVerification) {
  ModelScorer scorer;

  std::vector<ModelFeatures> models;
  for (int i = 0; i < 10; ++i) {
    ModelFeatures m;
    m.model_id = "model" + std::to_string(i);
    m.scores = {0.1f * (10 - i)};
    models.push_back(m);
  }

  auto scores = scorer.score_models(0, models);
  EXPECT_EQ(scores.size(), 10);

  for (size_t i = 1; i < scores.size(); ++i) {
    EXPECT_GE(scores[i - 1].score, scores[i].score)
        << "Scores not sorted descending: " << scores[i - 1].score << " < " << scores[i].score;
  }
}

TEST(ModelScorerTest, MoveConstructor) {
  ModelScorer scorer1;

  std::vector<ModelFeatures> models;
  ModelFeatures m1;
  m1.model_id = "model1";

  m1.scores = {0.1f};

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

  m1.scores = {0.1f};

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
